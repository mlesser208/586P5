import { useCallback, useEffect, useRef } from "react";
import mapboxgl, { GeoJSONSource, LngLatLike, Map as MapboxMap } from "mapbox-gl";

type FeatureCollection = GeoJSON.FeatureCollection<GeoJSON.Geometry, GeoJSON.GeoJsonProperties>;

type MapViewProps = {
  accessToken: string;
  /** Initial center for the map. Subsequent prop changes won't recreate the map. */
  center: LngLatLike;
  /** Initial zoom level. */
  zoom?: number;
  /** GeoJSON data that is applied imperatively without rebuilding the map. */
  data: FeatureCollection;
  /** Optional: called when the map finishes loading. */
  onLoad?: (map: MapboxMap) => void;
  /**
   * Optional: called when the viewport changes. Triggered on `moveend`/`zoomend`
   * to avoid firing every animation frame while the user is panning or zooming.
   */
  onMove?: (viewport: { center: [number, number]; zoom: number }) => void;
};

/**
 * React wrapper that initializes a Mapbox map once and updates layers imperatively.
 *
 * - The map instance is created a single time and stored in a ref.
 * - Data is applied to the existing source instead of remounting the map component.
 * - Viewport callbacks are throttled to avoid flooding React state.
 */
export function MapView({ accessToken, center, zoom = 11, data, onLoad, onMove }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapboxMap | null>(null);
  const moveFrame = useRef<number | null>(null);
  const initialCenter = useRef<LngLatLike>(center);
  const initialZoom = useRef<number>(zoom);
  const optionsRef = useRef({ accessToken, style: "mapbox://styles/mapbox/streets-v12" });
  const onLoadRef = useRef<MapViewProps["onLoad"]>(onLoad);
  const lastViewport = useRef<string | null>(null);

  // Keep the latest callback without forcing the map to re-initialize.
  useEffect(() => {
    onLoadRef.current = onLoad;
  }, [onLoad]);

  // Persist the initial options so parent re-renders don't retrigger map teardown.
  useEffect(() => {
    optionsRef.current = { ...optionsRef.current, accessToken };
  }, [accessToken]);

  // Create the map instance once.
  useEffect(() => {
    if (mapRef.current || !containerRef.current) return;

    mapboxgl.accessToken = optionsRef.current.accessToken;
    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: optionsRef.current.style,
      center: initialCenter.current,
      zoom: initialZoom.current,
      attributionControl: true,
      cooperativeGestures: true,
    });
    mapRef.current = map;

    map.on("load", () => {
      if (!map.getSource("listings")) {
        map.addSource("listings", {
          type: "geojson",
          data: { type: "FeatureCollection", features: [] },
        });
      }

      if (!map.getLayer("listings-points")) {
        map.addLayer({
          id: "listings-points",
          source: "listings",
          type: "circle",
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 8, 4, 14, 10],
            "circle-color": "#2563eb",
            "circle-stroke-color": "#0f172a",
            "circle-stroke-width": 1.25,
            "circle-opacity": 0.85,
          },
        });
      }

      onLoadRef.current?.(map);
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Imperatively update the GeoJSON source when data changes.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const applyData = () => {
      const source = map.getSource("listings") as GeoJSONSource | undefined;
      if (source) {
        source.setData(data);
      }
    };

    if (map.isStyleLoaded()) {
      applyData();
    } else {
      map.once("load", applyData);
    }
  }, [data]);

  // Fire move callbacks after zoom/pan completes, avoiding per-frame renders.
  const handleMove = useCallback(() => {
    const map = mapRef.current;
    if (!map || !onMove) return;

    if (moveFrame.current) cancelAnimationFrame(moveFrame.current);
    moveFrame.current = requestAnimationFrame(() => {
      const { lng, lat } = map.getCenter();
      const zoom = map.getZoom();
      const serialized = `${lng.toFixed(6)},${lat.toFixed(6)},${zoom.toFixed(3)}`;

      // Avoid spamming parent state with the same viewport values.
      if (serialized !== lastViewport.current) {
        lastViewport.current = serialized;
        onMove({ center: [lng, lat], zoom });
      }
    });
  }, [onMove]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !onMove) return;

    map.on("moveend", handleMove);
    map.on("zoomend", handleMove);
    return () => {
      map.off("moveend", handleMove);
      map.off("zoomend", handleMove);
      if (moveFrame.current) cancelAnimationFrame(moveFrame.current);
    };
  }, [handleMove, onMove]);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
