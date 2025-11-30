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
    Purpose: Typed stub for the streamlit_folium.st_folium helper, mirroring the
             runtime call signature we rely on in the Streamlit app.

    Parameters:
        figure (Any): Folium Map or Figure-like object to embed.
        width (int | None): Optional pixel width override for the rendered iframe.
        height (int | None): Optional pixel height override for the rendered iframe.
        returned_objects (Sequence[str] | None): Optional keys that instruct the
            widget which interactive results to return (e.g., clicks).
        feature_group_to_add (Iterable[Any] | None): Optional Folium feature groups
            that should be added to the map before rendering.
        component_id (str | None): Optional component identifier for Streamlit.
        key (str | None): Streamlit widget key.
        use_container_width (bool | None): Whether to expand to the parent width.
        kwargs (Any): Additional keyword arguments forwarded to the widget.

    Return Value:
        MutableMapping[str, Any]: A mapping containing interaction results such as
        details about the last clicked object or viewport bounds.

    Exceptions:
        Not raised in the stub; runtime errors are determined by the real implementation.
    """
    ...


