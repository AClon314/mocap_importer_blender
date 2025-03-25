# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "mocap-importer",
    "author": "Nolca",
    "description": "Import mocap data based on smpl-x model, support gvhmr, wilor, tram... Suggest to use with **mocap-wrapper**. 基于 smpl-x 模型导入 mocap 数据，支持 gvhmr、wilor、tram... 建议与**mocap-wrapper**一起使用。",
    "blender": (2, 80, 0),
    "version": (0, 2, 0),
    "location": "",
    "warning": "",
    "category": "Animation",
}
from . import auto_load
auto_load.init()


def register():
    auto_load.register()


def unregister():
    auto_load.unregister()
