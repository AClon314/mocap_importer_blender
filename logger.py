import bpy
import logging
from typing import Literal
MSG_TRUNK = 256
_PKG_ = __package__.split('.')[-1] if __package__ else __name__
_LOG_ICON = {
    'INFO': 'INFO',
    'WARNING': 'ERROR',
    'ERROR': 'CANCEL'
}
def Pop(dic: dict | None, key, default): return dic.pop(key, default) if dic else default


def msg_mouse(
    title="msg", msg="detail",
    icon: Literal['NONE', 'INFO', 'INFO_LARGE', 'WARNING_LARGE', 'ERROR', 'CANCEL'] | str = 'NONE'
):
    """`bpy.context.window_manager.popup_menu`"""

    def draw(self, context):
        self.layout.label(text=msg)
    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)  # type: ignore


class CustomLogger(logging.Logger):
    """Usage:
```python
extra={'mouse': True, 'report': True, 'log': True,
        'icon': ['NONE','INFO', 'INFO_LARGE', 'WARNING_LARGE', 'ERROR', 'CANCEL'][0], 
        'lvl': ['DEBUG','INFO','WARNING','ERROR',
                'ERROR_INVALID_INPUT','ERROR_INVALID_CONTEXT','ERROR_OUT_OF_MEMORY',
                'OPERATOR','PROPERTY'][0]
}"""

    def _log(self, level, msg: str, args, exc_info=None, extra: dict | None = None, stack_info=False, stacklevel=1):
        Level = logging.getLevelName(level)
        lvl: str = Pop(extra, 'lvl', Level)
        # stack_info = False if stack_info == None else stack_info
        is_mouse: bool = Pop(extra, 'mouse', level not in (logging.ERROR, logging.DEBUG))
        is_report: bool = Pop(extra, 'report', level > logging.INFO)
        is_log: bool = Pop(extra, 'log', not is_report)
        kwargs = dict(level=level, msg=msg, args=args, exc_info=exc_info, extra=extra, stack_info=stack_info, stacklevel=stacklevel)
        try:
            if is_mouse:
                icon: str = Pop(extra, 'icon', _LOG_ICON.get(Level, 'NONE'))    # type: ignore
                _msg = msg if len(msg) > MSG_TRUNK else _PKG_
                msg_mouse(title=msg[:MSG_TRUNK], msg=_msg, icon=icon)
            if is_report and hasattr(self, 'report'):
                self.report(type={lvl}, message=msg)   # type: ignore
            if is_log:
                super()._log(**kwargs)  # type: ignore
        except Exception as e:
            kwargs['msg'] = f"{kwargs['msg']}\tLogError: {e}"
            super()._log(**kwargs)  # type: ignore


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
                '%(levelname)s\t%(asctime)s %(message)s',  # %(name)s:%(lineno)d
                datefmt='%H:%M:%S'))
        Log.addHandler(stream_handler)
    return Log


Log = getLogger()
