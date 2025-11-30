from typing import Any, Iterable, Mapping, MutableMapping, Sequence


def st_folium(
    figure: Any,
    *,
    width: int | None = ...,
    height: int | None = ...,
    returned_objects: Sequence[str] | None = ...,
    feature_group_to_add: Iterable[Any] | None = ...,
    component_id: str | None = ...,
    key: str | None = ...,
    use_container_width: bool | None = ...,
    **kwargs: Any,
) -> MutableMapping[str, Any]:
    """
    Purpose: Provide static typing information for the streamlit_folium.st_folium
             helper to quiet type-checker warnings in this project.

    Parameters:
        figure (Any): Folium map or layer to render inside Streamlit.
        width (int | None): Optional width override for the embedded iframe.
        height (int | None): Optional height override for the embedded iframe.
        returned_objects (Sequence[str] | None): Optional keys describing which
            interactive artifacts to return (e.g., last_object_clicked).
        feature_group_to_add (Iterable[Any] | None): Optional Folium features to
            add to the map just-in-time before rendering.
        component_id (str | None): Optional Streamlit component identifier.
        key (str | None): Streamlit widget key to preserve state.
        use_container_width (bool | None): Whether to expand to the parent width.
        kwargs (Any): Additional keyword args forwarded to the component.

    Return Value:
        MutableMapping[str, Any]: Dictionary-like object with interactive outputs
        such as click metadata or bounds.

    Exceptions:
        Not raised in stub definitions; runtime component controls actual errors.
    """
    ...


