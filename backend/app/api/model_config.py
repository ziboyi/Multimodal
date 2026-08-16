from fastapi import APIRouter, Depends
from app.core.dependencies import get_current_user_id
from app.schemas.model_config import (
    ModelConfigCreate,
    ModelConfigUpdate,
    ModelConfigResponse,
)

router = APIRouter(prefix="/models", tags=["模型配置"])


@router.post("", response_model=ModelConfigResponse)
async def create_model_config(
    req: ModelConfigCreate,
    user_id: str = Depends(get_current_user_id),
):
    """创建模型配置"""
    # TODO: 实现
    pass


@router.get("", response_model=list[ModelConfigResponse])
async def list_model_configs(
    user_id: str = Depends(get_current_user_id),
):
    """获取模型配置列表"""
    # TODO: 实现
    pass


@router.patch("/{config_id}", response_model=ModelConfigResponse)
async def update_model_config(
    config_id: str,
    req: ModelConfigUpdate,
    user_id: str = Depends(get_current_user_id),
):
    """更新模型配置"""
    # TODO: 实现
    pass


@router.delete("/{config_id}")
async def delete_model_config(
    config_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """删除模型配置"""
    # TODO: 实现
    pass
