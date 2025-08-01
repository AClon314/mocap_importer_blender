from .FKtoIK import fktoikaddon
from ..logger import Log
from ..lib import Progress, GEN
fktoikaddon.Log = Log
fktoikaddon.Progress = Progress
fktoikaddon.GEN = GEN
# IS_DEBUG = bpy.context.preferences.view.show_developer_ui
from .FKtoIK.fktoikaddon import fk_to_ik
REG = [
    fktoikaddon.register
]
UNREG = [
    fktoikaddon.unregister
]
def register(): [reg() for reg in REG]
def unregister(): [unreg() for unreg in UNREG]
