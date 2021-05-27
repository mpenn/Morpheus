from collections import defaultdict
import inspect
import typing

T_co = typing.TypeVar("T_co", covariant=True)
T = typing.TypeVar('T')
T1 = typing.TypeVar('T1')
T2 = typing.TypeVar('T2')
T3 = typing.TypeVar('T3')
T4 = typing.TypeVar('T4')

def greatest_ancestor(*cls_list):
    mros = [list(inspect.getmro(cls)) for cls in cls_list]
    track = defaultdict(int)
    while mros:
        for mro in mros:
            cur = mro.pop(0)
            track[cur] += 1
            if track[cur] == len(cls_list):
                return cur
            if len(mro) == 0:
                mros.remove(mro)
    return None  # or raise, if that's more appropriate

@typing.overload
def unpack_union(cls_1: typing.Type[T]) -> typing.Union[typing.Type[T]]:
    ...

@typing.overload
def unpack_union(cls_1: typing.Type[T1], cls_2: typing.Type[T2]) -> typing.Union[typing.Type[T1], typing.Type[T2]]:
    ...

@typing.overload
def unpack_union(cls_1: typing.Type[T1], cls_2: typing.Type[T2], cls_3: typing.Type[T3]) -> typing.Union[typing.Type[T1], typing.Type[T2], typing.Type[T3]]:
    ...

def unpack_union(*cls_list: typing.Type) -> typing.Union:

    assert len(cls_list) > 0, "Union class list must have at least 1 element."

    if (len(cls_list) == 1):
        return typing.Union[cls_list[0]]
    # elif (len(cls_list) == 2):
    #     return typing.Union[cls_list[0], cls_list[1]]
    else:
        out_union = unpack_union(cls_list[0:2])

        # Since typing.Union[typing.Union[A, B], C] == typing.Union[A, B, C], we build the union up manually
        for t in cls_list[2:]:
            out_union = typing.Union[out_union, t]

        return out_union


@typing.overload
def unpack_tuple(cls_1: typing.Type[T]) -> typing.Tuple[typing.Type[T]]:
    ...

@typing.overload
def unpack_tuple(cls_1: typing.Type[T1], cls_2: typing.Type[T2]) -> typing.Tuple[typing.Type[T1], typing.Type[T2]]:
    ...

@typing.overload
def unpack_tuple(cls_1: typing.Type[T1], cls_2: typing.Type[T2], cls_3: typing.Type[T3]) -> typing.Tuple[typing.Type[T1], typing.Type[T2], typing.Type[T3]]:
    ...

def unpack_tuple(*cls_list: typing.Type) -> typing.Tuple:

    assert len(cls_list) > 0, "Union class list must have at least 1 element."

    if (len(cls_list) == 1):
        return typing.Tuple[cls_list[0]]
    # elif (len(cls_list) == 2):
    #     return typing.Union[cls_list[0], cls_list[1]]
    else:
        out_tuple = unpack_tuple(cls_list[0:2])

        # Since typing.Tuple[typing.Tuple[A, B], C] == typing.Tuple[A, B, C], we build the union up manually
        for t in cls_list[2:]:
            out_tuple = typing.Tuple[out_tuple, t]

        return out_tuple