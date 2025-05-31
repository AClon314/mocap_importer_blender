# mocap_importer

## Suggest
- [mocap-wrapper](https://github.com/AClon314/mocap-wrapper) to install/get mocap data.
- [smpl-x blender addon](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_blender_addon_lh_20241129.zip) or [source Code](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon) to import smplx model.

Currently support gvhmr as `.npz` file.

![screenshot](doc/addon.png)

## TODO
PR welcome! (ゝ∀･)

- only import selected bones
- remember which .npz for each armature

draft,done partly, not perfect:
- bones remapping to UE Mannequin (external roroko addon)
  - auto T-pose by mesh, then apply modifier with keeping shape key, then rokoko retargeting
- bbox viewer
- [track camera from gvhmr](https://github.com/zju3dv/GVHMR/issues/30)
- make wilor predict hands ID continuously
- Import all motion:
  - at 0 frame/at begin_frame(described in npz)
  - on same person/ auto new person