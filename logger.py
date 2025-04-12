import bpy
import logging
from contextlib import contextmanager
from typing import Optional, Sequence, Literal
MSG_TRUNK = 256
_PKG_ = __package__.split('.')[-1] if __package__ else __name__
_LOG_ICON = {
    'INFO': 'INFO',
    'WARNING': 'ERROR',
    'ERROR': 'CANCEL'
}
def Pop(dic: Optional[dict], key, default): return dic.pop(key, default) if dic else default


def msg_mouse(
    title="msg", msg="detail",
    icon: Literal['NONE', 'INFO', 'INFO_LARGE', 'WARNING_LARGE', 'ERROR', 'CANCEL'] | str = 'NONE'
):
    """`bpy.context.window_manager.popup_menu`"""

    def draw(self, context):
        self.layout.label(text=msg)
    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)  # type: ignore


class Report_Operator(bpy.types.Operator):
    """deprecated, only working when trigger from GUI`F3`, can't push to INFO LOG PANEL: https://projects.blender.org/blender/blender/issues/97284  
    report to status bar, https://docs.blender.org/api/current/bpy_types_enum_items/wm_report_items.html#rna-enum-wm-report-items"""
    bl_idname = 'wm.report'
    bl_label = f'Report Message ({_PKG_})'
    msg: bpy.props.StringProperty(name="Message", default="Log operator msg")  # type: ignore
    lvl: bpy.props.StringProperty(name="Level", default='INFO')    # type: ignore

    def execute(self, context):
        msg = f'{self.msg} {context} {len(dir(context))}'
        self.report({self.lvl}, msg)
        return {'FINISHED'}


class CustomLogger(logging.Logger):
    """```python
extra={'mouse': True, 'report': True, 
    'icon': ['NONE','INFO', 'INFO_LARGE', 'WARNING_LARGE', 'ERROR', 'CANCEL'][0], 
    'lvl': ['DEBUG','INFO','WARNING','ERROR',
            'ERROR_INVALID_INPUT','ERROR_INVALID_CONTEXT','ERROR_OUT_OF_MEMORY',
            'OPERATOR','PROPERTY'][0]
}"""

    def _log(self, level, msg: str, args, exc_info=None, extra: Optional[dict] = None, stack_info=False, stacklevel=2):
        Level = logging.getLevelName(level)
        lvl: str = Pop(extra, 'lvl', Level)
        is_mouse: bool = Pop(extra, 'mouse', True)
        is_report: bool = Pop(extra, 'report', False)
        kwargs = dict(level=level, msg=msg, args=args, exc_info=exc_info, extra=extra, stack_info=stack_info, stacklevel=stacklevel)
        try:
            if is_mouse:
                icon: str = Pop(extra, 'icon', _LOG_ICON.get(Level, 'NONE'))    # type: ignore
                _msg = msg if len(msg) > MSG_TRUNK else ''
                msg_mouse(title=msg[:MSG_TRUNK], msg=_msg, icon=icon)
            if is_report:
                bpy.ops.wm.report(lvl=lvl, msg=msg)  # type: ignore
            else:
                super()._log(**kwargs)
        except Exception as e:
            kwargs['msg'] = f"{kwargs['msg']}\tLog:{e}"
            super()._log(**kwargs)


def getLogger(name=__name__, level=logging.DEBUG):
    """```python
    Log = getLogger(__name__)
    ```"""
    logging.setLoggerClass(CustomLogger)
    Log = logging.getLogger(name)
    Log.setLevel(level)
    Log.propagate = False  # 禁止传播到根Logger

    # 避免重复添加handler
    if not Log.handlers:
        stream_handler = logging.StreamHandler()

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                return super().format(record)

        stream_handler.setFormatter(
            CustomFormatter(
                '%(levelname)s\t%(asctime)s  %(message)s\t%(name)s:%(lineno)d',
                datefmt='%H:%M:%S'))
        Log.addHandler(stream_handler)

    return Log


@contextmanager
def progress_mouse(Range: Sequence[float] = (0, 10000, 1)):
    """上下文管理器，显示进度条"""
    wm = bpy.context.window_manager
    if not wm:
        raise RuntimeError("No window manager found")
    wm.progress_begin(Range[0], Range[1])
    try:
        yield wm.progress_update
    finally:
        wm.progress_end()


class ModalTimerOperator(bpy.types.Operator):
    """https://blender.stackexchange.com/a/305675/146607"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"
    _timer = None

    def modal(self, context, event):
        [a.tag_redraw() for a in context.screen.areas]
        if self._timer.time_duration > 3:
            context.window_manager.progress = 1
            return {'FINISHED'}
        context.window_manager.progress = self._timer.time_duration / 3
        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}
