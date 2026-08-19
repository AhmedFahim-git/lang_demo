from typing import Any, TypedDict

from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen, MultipleOf
from pydantic import BaseModel, Field, Strict, create_model
from pydantic.fields import FieldInfo


class FieldParams(TypedDict, total=False):
    description: str
    default: str | int | float | bool
    strict: bool
    gt: int | float
    ge: int | float
    lt: int | float
    le: int | float
    multiple_of: int | float
    allow_inf_nan: bool
    min_length: int
    max_length: int


STR_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}

TYPE_STR_MAP: dict[type, str] = {v: k for k, v in STR_TYPE_MAP.items()}

METADATA_TYPE_STR: dict[Any, str] = {
    Ge: "ge",
    Gt: "gt",
    Le: "le",
    Lt: "lt",
    MultipleOf: "multiple_of",
    MinLen: "min_length",
    MaxLen: "max_length",
    Strict: "strict",
}


class PydanticFieldDef(BaseModel):
    name: str
    type: str
    field_params: FieldParams


class PydanticModelDef(BaseModel):
    name: str
    fields: list[PydanticFieldDef]


def make_pydantic_model_from_def(model_def_str: str):
    model_def = PydanticModelDef.model_validate_json(model_def_str)
    fields: dict[str, Any] = {
        field_def.name: (STR_TYPE_MAP[field_def.type], Field(**field_def.field_params))
        for field_def in model_def.fields
    }
    return create_model(model_def.name, **fields)


def get_model_def(
    req_fields: dict[str, dict[str, str | int | float | bool | None]],
    fields_dict: dict[str, FieldInfo],
    name: str,
) -> PydanticModelDef:
    fields: list[PydanticFieldDef] = []
    for field, in_attrs in req_fields.items():
        field_info = fields_dict[field]
        field_info_dict = field_info.asdict()
        attributes = field_info_dict.get("attributes")
        metadata = field_info_dict.get("metadata")
        metadata_dict: dict[str, str | int | float | bool] = {}
        for item in metadata:
            if type(item) in METADATA_TYPE_STR:
                metadata_str = METADATA_TYPE_STR[type(item)]
                metadata_dict[metadata_str] = getattr(item, metadata_str)
            elif hasattr(item, "allow_inf_nan"):
                metadata_dict["allow_inf_nan"] = item.allow_inf_nan

        if field_info.is_required():
            attributes["default"] = None
        attributes.update(metadata_dict)
        attributes.update(in_attrs)
        field_def: FieldParams = FieldParams(
            **{
                k: attributes[k]
                for k in FieldParams.__annotations__
                if attributes.get(k) is not None
            }
        )
        assert field_info.annotation is not None
        fields.append(
            PydanticFieldDef(
                name=field,
                type=TYPE_STR_MAP[field_info.annotation],
                field_params=field_def,
            )
        )
    return PydanticModelDef(name=name, fields=fields)
