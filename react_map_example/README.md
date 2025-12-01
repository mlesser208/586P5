# React Map Lifecycle Example

This example shows how to create a Mapbox GL JS map **once** in React and update its data without forcing re-renders of the map container. The pattern mirrors how Apple Maps or Zillow feel responsive while minimizing unnecessary rebuilds.

## Key ideas implemented
- **Single map instance:** The Mapbox map is created a single time with `useRef`. Changing props in React does not recreate the map container, and callbacks are stored in refs so passing a new `onLoad` function will not tear down the map.
- **Imperative data updates:** GeoJSON data is pushed directly into an existing source (`source.setData(...)`) instead of remounting the map or layers.
- **Isolated lifecycle:** Map creation/removal happens in one effect; data updates live in a separate effect so UI state changes do not touch the map instance.
- **Throttled callbacks:** Viewport change callbacks fire on `moveend/zoomend` with `requestAnimationFrame`, so your React state updates only after a pan or zoom settles (not every animation frame).
- **Stable options:** Initial `center`/`zoom` are captured once so parent re-renders do not reset the camera.

## Usage
1. Install Mapbox GL JS:

```bash
npm install mapbox-gl
```

2. Import and render the component:

```tsx
import type { FeatureCollection } from "geojson";
import { MapView } from "./react_map_example/MapView";

const data: FeatureCollection = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [-118.2437, 34.0522] },
      properties: { id: "la", label: "Los Angeles" },
    },
  ],
};

export function App() {
  return (
    <div style={{ width: "100%", height: "600px" }}>
      <MapView
        accessToken={process.env.VITE_MAPBOX_TOKEN!}
        center={[-118.2437, 34.0522]}
        zoom={11}
        data={data}
        onMove={(viewport) => {
          // Persist viewport in state if you want, without re-rendering the map.
          console.log("viewport", viewport);
        }}
      />
    </div>
  );
}
```

3. Update `data` when listings change. Because the component updates the existing GeoJSON source imperatively, the map does **not** remount or re-render.

## Tips to avoid extra renders
- Keep the map container mounted once; do not conditionally render it based on unrelated UI state.
- Memoize expensive data transformations before passing them into `<MapView />`.
- Debounce or throttle network requests triggered by viewport changes (e.g., wait for `moveend` or throttle the `onMove` callback as shown).
- Only update GeoJSON when the underlying data actually changes; compare hashes or timestamps if needed.
- If you need markers instead of a GeoJSON layer, keep marker objects in refs and add/remove them imperatively just like the GeoJSON source.
- If you pass callbacks like `onLoad` or `onMove`, prefer stable references (e.g., `useCallback`) so parent re-renders do not trigger unnecessary React work. The map itself keeps the latest callbacks via refs and does not rebuild.
