from nerfstudio.data.scene_box import SceneBox
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from nerfstudio.models.base_model import Model, ModelConfig

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, geom_init=False, weight_norm=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.geom_init = geom_init

        net = []
        for l in range(num_layers):

            in_dim = self.dim_in if l == 0 else self.dim_hidden
            out_dim = self.dim_out if l == num_layers - 1 else self.dim_hidden

            net.append(nn.Linear(in_dim, out_dim, bias=bias))
        
            if geom_init:
                if l == num_layers - 1:
                    torch.nn.init.normal_(net[l].weight, mean=math.sqrt(math.pi) / math.sqrt(in_dim), std=1e-4)
                    if bias: torch.nn.init.constant_(net[l].bias, -0.5) # sphere init (very important for hashgrid encoding!)

                elif l == 0:
                    torch.nn.init.normal_(net[l].weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(out_dim))
                    torch.nn.init.constant_(net[l].weight[:, 3:], 0.0)
                    if bias: torch.nn.init.constant_(net[l].bias, 0.0)

                else:
                    torch.nn.init.normal_(net[l].weight, 0.0, math.sqrt(2) / math.sqrt(out_dim))
                    if bias: torch.nn.init.constant_(net[l].bias, 0.0)
            
            if weight_norm:
                net[l] = nn.utils.weight_norm(net[l])

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                if self.geom_init:
                    x = F.softplus(x, beta=100)
                else:
                    x = F.relu(x, inplace=True)
        return x
    

class NeRFRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # bound for ray marching (world space)
        self.real_bound = opt.bound

        # bound for grid querying
        if self.opt.contract:
            self.bound = 2
        else:
            self.bound = opt.bound

        self.cascade = 1 + math.ceil(math.log2(self.bound))

        self.grid_size = opt.grid_size
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh

        self.max_level = 16

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor(
            [
                -self.real_bound,
                -self.real_bound,
                -self.real_bound,
                self.real_bound,
                self.real_bound,
                self.real_bound,
            ]
        )
        aabb_infer = aabb_train.clone()
        self.register_buffer("aabb_train", aabb_train)
        self.register_buffer("aabb_infer", aabb_infer)
        

        # individual codes
        self.individual_num = opt.ind_num
        self.individual_dim = opt.ind_dim

        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(
                torch.randn(self.individual_num, self.individual_dim) * 0.1
            )
        else:
            self.individual_codes = None

        # extra state for cuda raymarching
        self.cuda_ray = opt.cuda_ray
        assert self.cuda_ray

        # density grid
        if not self.opt.trainable_density_grid:
            density_grid = torch.zeros(
                [self.cascade, self.grid_size**3]
            )  # [CAS, H * H * H]
            self.register_buffer("density_grid", density_grid)
        else:
            self.density_grid = nn.Parameter(
                torch.zeros([self.cascade, self.grid_size**3])
            )  # [CAS, H * H * H]
        density_bitfield = torch.zeros(
            self.cascade * self.grid_size**3 // 8, dtype=torch.uint8
        )  # [CAS * H * H * H // 8]
        self.register_buffer("density_bitfield", density_bitfield)
        self.mean_density = 0
        self.iter_density = 0

        # for second phase training

        if self.opt.stage == 1:
            if self.opt.gui:
                self.glctx = (
                    dr.RasterizeCudaContext()
                )  # support at most 2048 resolution.
            else:
                self.glctx = dr.RasterizeGLContext(
                    output_db=False
                )  # will crash if using GUI...

            # sequentially load cascaded meshes
            vertices = []
            triangles = []
            v_cumsum = [0]
            f_cumsum = [0]
            for cas in range(self.cascade):
                _updated_mesh_path = (
                    os.path.join(
                        self.opt.workspace, "mesh_stage0", f"mesh_{cas}_updated.ply"
                    )
                    if self.opt.mesh == ""
                    else self.opt.mesh
                )
                if os.path.exists(_updated_mesh_path) and self.opt.ckpt != "scratch":
                    mesh = trimesh.load(
                        _updated_mesh_path,
                        force="mesh",
                        skip_material=True,
                        process=False,
                    )
                else:  # base (not updated)
                    mesh = trimesh.load(
                        os.path.join(
                            self.opt.workspace, "mesh_stage0", f"mesh_{cas}.ply"
                        ),
                        force="mesh",
                        skip_material=True,
                        process=False,
                    )
                print(
                    f"[INFO] loaded cascade {cas} mesh: {mesh.vertices.shape}, {mesh.faces.shape}"
                )

                vertices.append(mesh.vertices)
                triangles.append(mesh.faces + v_cumsum[-1])

                v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
                f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])

            vertices = np.concatenate(vertices, axis=0)
            triangles = np.concatenate(triangles, axis=0)
            self.v_cumsum = np.array(v_cumsum)
            self.f_cumsum = np.array(f_cumsum)

            # must put to cuda manually, we don't want these things in the model as buffers...
            self.vertices = torch.from_numpy(vertices).float().cuda()  # [N, 3]
            self.triangles = torch.from_numpy(triangles).int().cuda()

            # learnable offsets for mesh vertex
            self.vertices_offsets = nn.Parameter(torch.zeros_like(self.vertices))

            # accumulate error for mesh face
            self.triangles_errors = torch.zeros_like(
                self.triangles[:, 0], dtype=torch.float32
            ).cuda()
            self.triangles_errors_cnt = torch.zeros_like(
                self.triangles[:, 0], dtype=torch.float32
            ).cuda()
            self.triangles_errors_id = None

        else:
            self.glctx = None

    def get_params(self, lr):
        params = []

        if self.individual_codes is not None:
            params.append(
                {
                    "params": self.individual_codes,
                    "lr": self.opt.lr * 0.1,
                    "weight_decay": 0,
                }
            )

        if self.opt.trainable_density_grid:
            params.append(
                {"params": self.density_grid, "lr": self.opt.lr, "weight_decay": 0}
            )

        if self.glctx is not None:
            params.append(
                {
                    "params": self.vertices_offsets,
                    "lr": self.opt.lr_vert,
                    "weight_decay": 0,
                }
            )

        return params


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 specular_dim=3,
                 ):

        super().__init__(opt)

        # density network
        self.encoder, self.in_dim_density = get_encoder("hashgrid_tcnn" if self.opt.tcnn else "hashgrid", level_dim=1, desired_resolution=2048 * self.bound, interpolation='linear')
        # self.sigma_net = MLP(3 + self.in_dim_density, 1, 32, 2, bias=self.opt.sdf, geom_init=self.opt.sdf, weight_norm=self.opt.sdf)
        self.sigma_net = MLP(3 + self.in_dim_density, 1, 32, 2, bias=False)

        # color network
        self.encoder_color, self.in_dim_color = get_encoder("hashgrid_tcnn" if self.opt.tcnn else "hashgrid", level_dim=2, desired_resolution=2048 * self.bound, interpolation='linear')
        self.color_net = MLP(3 + self.in_dim_color + self.individual_dim, 3 + specular_dim, 64, 3, bias=False)

        self.encoder_dir, self.in_dim_dir = get_encoder("None")
        self.specular_net = MLP(specular_dim + self.in_dim_dir, 3, 32, 2, bias=False)

        # sdf
        if self.opt.sdf:
            self.register_parameter('variance', nn.Parameter(torch.tensor(0.3, dtype=torch.float32)))

    def forward(self, x, d, c=None, shading='full'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # c: [1/N, individual_dim]

        sigma = self.density(x)['sigma']
        color, specular = self.rgb(x, d, c, shading)

        return sigma, color, specular


    def density(self, x):

        # sigma
        h = self.encoder(x, bound=self.bound, max_level=self.max_level)
        h = torch.cat([x, h], dim=-1)
        h = self.sigma_net(h)

        results = {}

        if self.opt.sdf:
            sigma = h[..., 0].float() # sdf
        else:
            sigma = trunc_exp(h[..., 0])

        results['sigma'] = sigma

        return results

    # init the sdf to two spheres by pretraining, assume view cameras fall between the spheres
    def init_double_sphere(self, r1=0.5, r2=1.5, iters=8192, batch_size=8192):
        # sphere init is only for sdf mode!
        if not self.opt.sdf:
            return
        # import kiui
        import tqdm
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)
        pbar = tqdm.trange(iters, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for _ in range(iters):
            # random points inside [-b, b]^3
            xyzs = torch.rand(batch_size, 3, device='cuda') * 2 * self.bound - self.bound
            d = torch.norm(xyzs, p=2, dim=-1)
            gt_sdf = torch.where(d < (r1 + r2) / 2, d - r1, r2 - d)
            # kiui.lo(xyzs, gt_sdf)
            pred_sdf = self.density(xyzs)['sigma']
            loss = loss_fn(pred_sdf, gt_sdf)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f'pretrain sdf loss={loss.item():.8f}')
            pbar.update(1)
    
    # finite difference
    def normal(self, x, epsilon=1e-4):

        if self.opt.tcnn:
            with torch.enable_grad():
                x.requires_grad_(True)
                sigma = self.density(x)['sigma']
                normal = torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]
        else:
            dx_pos = self.density((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dx_neg = self.density((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dy_pos = self.density((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dy_neg = self.density((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dz_pos = self.density((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dz_neg = self.density((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            
            normal = torch.stack([
                0.5 * (dx_pos - dx_neg) / epsilon, 
                0.5 * (dy_pos - dy_neg) / epsilon, 
                0.5 * (dz_pos - dz_neg) / epsilon
            ], dim=-1)

        return normal
    

    def geo_feat(self, x, c=None):

        h = self.encoder_color(x, bound=self.bound, max_level=self.max_level)
        h = torch.cat([x, h], dim=-1)
        if c is not None:
            h = torch.cat([h, c.repeat(x.shape[0], 1) if c.shape[0] == 1 else c], dim=-1)
        h = self.color_net(h)
        geo_feat = torch.sigmoid(h)

        return geo_feat


    def rgb(self, x, d, c=None, shading='full'):

        # color
        geo_feat = self.geo_feat(x, c)
        diffuse = geo_feat[..., :3]

        if shading == 'diffuse':
            color = diffuse
            specular = None
        else: 
            d = self.encoder_dir(d)
            specular = self.specular_net(torch.cat([d, geo_feat[..., 3:]], dim=-1))
            specular = torch.sigmoid(specular)
            if shading == 'specular':
                color = specular
            else: # full
                color = (specular + diffuse).clamp(0, 1) # specular + albedo

        return color, specular


    # optimizer utils
    def get_params(self, lr):

        params = super().get_params(lr)

        params.extend([
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.encoder_color.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
            {'params': self.specular_net.parameters(), 'lr': lr}, 
        ])

        if self.opt.sdf:
            params.append({'params': self.variance, 'lr': lr * 0.1})

        return params
    

class NeRF2Mesh(Model):
    def __init__(self, config: ModelConfig, scene_box: SceneBox, num_train_data: int, **kwargs) -> None:
        super().__init__(config, scene_box, num_train_data, **kwargs)
        ...