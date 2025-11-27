from typing import Any, Mapping, MutableMapping, Sequence, TypeAlias


DataFrame: TypeAlias = Any
Series: TypeAlias = Any


class _Errors:
    """
    Purpose: Container for pandas.errors exception stubs used within the project.

    Parameters:
        None.

    Return Value:
        Not applicable.

    Exceptions:
        None.
    """

    class EmptyDataError(Exception):
        """
        Purpose: Stub exception mirroring pandas.errors.EmptyDataError.

        Parameters:
            *args (Any): Positional arguments forwarded to Exception.
            **kwargs (Any): Keyword arguments forwarded to Exception.

        Return Value:
            None.

        Exceptions:
            Inherits from Exception.
        """


errors: _Errors


def read_csv(
    filepath_or_buffer: str | bytes,
    *,
    low_memory: bool | None = ...,
    dtype: Mapping[str, Any] | None = ...,
    usecols: Sequence[str] | None = ...,
    encoding: str | None = ...,
    na_values: Sequence[str] | None = ...,
    keep_default_na: bool = ...,
    **kwargs: Any,
) -> DataFrame:
    """
    Purpose: Stub signature for pandas.read_csv used when loading housing data.

    Parameters:
        filepath_or_buffer (str | bytes): Path or buffer to read.
        low_memory (bool | None): Whether to use pandas' low-memory mode.
        dtype (Mapping[str, Any] | None): Optional column dtypes.
        usecols (Sequence[str] | None): Optional columns to restrict reading.
        encoding (str | None): Encoding override.
        na_values (Sequence[str] | None): Additional NA tokens.
        keep_default_na (bool): Whether to keep pandas' default NA tokens.
        kwargs (Any): Other read_csv keyword arguments.

    Return Value:
        DataFrame: Parsed tabular data.

    Exceptions:
        Raises FileNotFoundError or parsing-related errors at runtime.
    """
    ...


def to_numeric(arg: Any, *, errors: str = ..., downcast: str | None = ...) -> Series:
    """
    Purpose: Stub signature for pandas.to_numeric used to coerce numeric columns.

    Parameters:
        arg (Any): The data to convert to numeric values.
        errors (str): Error handling policy ("raise", "coerce", "ignore").
        downcast (str | None): Optional downcast instruction.

    Return Value:
        Series: Converted numeric data.

    Exceptions:
        ValueError: Raised when conversion fails and errors="raise".
    """
    ...


def notna(obj: Any) -> bool:
    """
    Purpose: Stub signature for pandas.notna used to test for non-missing values.

    Parameters:
        obj (Any): Scalar or array-like to test.

    Return Value:
        bool: True when the value is not NA.

    Exceptions:
        None (matches pandas.notna behavior).
    """
    ...

