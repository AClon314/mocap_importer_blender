from .FKtoIK import fktoikaddon
# IS_DEBUG = bpy.context.preferences.view.show_developer_ui
REG = [
    # fktoikaddon.register
]
UNREG = [
    # fktoikaddon.unregister
]
def register(): [reg() for reg in REG]
def unregister(): [unreg() for unreg in UNREG]
