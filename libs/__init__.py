from ..logger import Log
from ..lib import Progress, GEN

# IS_DEBUG = bpy.context.preferences.view.show_developer_ui
try:
    from .FKtoIK import fktoikaddon

    fktoikaddon.Log = Log
    fktoikaddon.Progress = Progress
    fktoikaddon.GEN = GEN
    from .FKtoIK.fktoikaddon import gen_fk_to_ik

    REG = [fktoikaddon.register]
    UNREG = [fktoikaddon.unregister]

    def register():
        [reg() for reg in REG]

    def unregister():
        [unreg() for unreg in UNREG]

except ImportError:
    import os

    Log.warning(
        f"Get FKtoIK at https://github.com/AClon314/FKtoIK/tree/main, then place under {os.path.join(os.path.dirname(__file__), 'FKtoIK')}"
    )
