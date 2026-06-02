"""Isolate the bug: compare pos_pairs between arithmetic and broadcast Phase1a."""
import numpy as np, sys, torch
sys.path.insert(0, '.')
from inference.pipeline.corrector import Corrector, CorrectorConfig
from inference.pipeline.pbc import build_ghost_tile, min_nn_pbc
from inference.pipeline.tiling import iter_tiles
from data.data_processor import DataProcessor

cfg = CorrectorConfig.from_yaml('inference/configs/grid_6x6.yaml')
cfg.device = 'cpu'
c = Corrector(cfg)
pos = np.load('inference/sph_data/positions_without.npy')[300].astype('float32')
G=c._G; cell=c._cell; ghost_w=c._ghost_w; dom=c._dom; n_tiles=G*G; N=len(pos)

# ── Arithmetic Phase1a (new) ──────────────────────────────────────────────────
any_in_ext_new = np.zeros((n_tiles,N),dtype=bool)
which_img_new  = np.zeros((n_tiles,N),dtype=np.int8)
is_core_new    = np.zeros((n_tiles,N),dtype=bool)
for k_img,(ox,oy) in enumerate(c._img_offsets):
    sx=pos[:,0]+ox; sy=pos[:,1]+oy
    ti_f=np.floor(sx/cell).astype(np.int32); tj_f=np.floor(sy/cell).astype(np.int32)
    xr=sx-ti_f*cell; yr=sy-tj_f*cell
    lo_x=xr<ghost_w; hi_x=xr>=cell-ghost_w; lo_y=yr<ghost_w; hi_y=yr>=cell-ghost_w
    for dti,mx in((-1,lo_x),(0,None),(1,hi_x)):
        for dtj,my in((-1,lo_y),(0,None),(1,hi_y)):
            if mx is None and my is None: p_in=np.arange(N,dtype=np.int32)
            elif mx is None: p_in=np.where(my)[0].astype(np.int32)
            elif my is None: p_in=np.where(mx)[0].astype(np.int32)
            else: p_in=np.where(mx&my)[0].astype(np.int32)
            if not p_in.size: continue
            t_in=((ti_f[p_in]+dti)%G)*G+(tj_f[p_in]+dtj)%G
            new=~any_in_ext_new[t_in,p_in]
            if new.any():
                any_in_ext_new[t_in[new],p_in[new]]=True
                which_img_new[t_in[new],p_in[new]]=k_img
            if k_img==4 and dti==0 and dtj==0:
                is_core_new[t_in,p_in]=True

t_idx_new,p_idx_new=np.where(any_in_ext_new)
ox_arr=np.array([o[0] for o in c._img_offsets],dtype=np.float32)
oy_arr=np.array([o[1] for o in c._img_offsets],dtype=np.float32)
k_arr=which_img_new[t_idx_new,p_idx_new].astype(np.int32)
pos_pairs_new=np.stack([pos[p_idx_new,0]+ox_arr[k_arr],
                        pos[p_idx_new,1]+oy_arr[k_arr]],axis=1)

# ── Broadcast Phase1a (old, the version before arithmetic) ────────────────────
offsets=np.array([(dx*dom,dy*dom) for dx in(-1,0,1) for dy in(-1,0,1)],dtype=np.float32)
shifted_all=pos[None]+offsets[:,None]   # (9,N,2)
in_ext=np.all(
    (shifted_all[:,None]>=c._ext_lo[None,:,None])&
    (shifted_all[:,None]< c._ext_hi[None,:,None]),axis=3)  # (9,n_tiles,N)
any_in_ext_old=in_ext.any(axis=0)
in_core_old=np.all(
    (pos[None]>=c._tile_lo[:,None])&(pos[None]<c._tile_hi[:,None]),axis=2)
is_core_old=in_ext[4]&in_core_old
which_img_old=in_ext.argmax(axis=0)

t_idx_old,p_idx_old=np.where(any_in_ext_old)
pos_pairs_old=np.stack([pos[p_idx_old,0]+offsets[which_img_old[t_idx_old,p_idx_old],0],
                         pos[p_idx_old,1]+offsets[which_img_old[t_idx_old,p_idx_old],1]],axis=1)

# ── Compare membership and pos_pairs ─────────────────────────────────────────
print(f'any_in_ext match:  {np.array_equal(any_in_ext_new, any_in_ext_old)}')
print(f'is_core match:     {np.array_equal(is_core_new, is_core_old)}')
print(f'which_img match:   {np.array_equal(which_img_new, which_img_old)}')

# Compare pos_pairs for each (t,p) pair (need to match by (t,p) index)
# Both should have same (t_idx,p_idx) in same order since np.where returns sorted
same_tp = np.array_equal(
    np.stack([t_idx_new,p_idx_new],1),
    np.stack([t_idx_old,p_idx_old],1))
print(f'(t_idx,p_idx) same order: {same_tp}')
if same_tp:
    pp_diff = np.abs(pos_pairs_new - pos_pairs_old)
    print(f'pos_pairs max diff: {pp_diff.max():.2e}')
    n_bad = (pp_diff.max(axis=1) > 1e-5).sum()
    print(f'pos_pairs pairs with diff>1e-5: {n_bad} / {len(pos_pairs_new)}')
