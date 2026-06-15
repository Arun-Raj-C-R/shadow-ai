import os
import json
import webbrowser
import time
from datetime import datetime
from google import genai
from google.genai import types

# Use the same API Key as code5.py
API_KEY = "AIzaSyDq5lgR3TbmMejCQfgLje9oIGbTvuDnWiM"

client = genai.Client(api_key=API_KEY)

# Rate limits (RPM, TPM, RPD) as requested by user
# Primary: Gemini 3.1 Flash Lite
# Fallback: Gemini 2.5 Flash
MODEL_FALLBACKS = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash"
]

# Track usage (simplified for this session)
usage_stats = {"rpm": 0, "tpm": 0, "rpd": 0, "last_reset": time.time()}

def check_limits():
    """Reset counters if minute/day has passed (simplified)."""
    now = time.time()
    if now - usage_stats["last_reset"] > 60:
        usage_stats["rpm"] = 0
        usage_stats["tpm"] = 0
        usage_stats["last_reset"] = now

def run_google_maps_tool(prompt, lat=None, lng=None, zoom=12, task="find_spots", route_from=None, route_to=None):
    """
    Executes a Gemini prompt with Google Maps grounding and opens an interactive visualization.
    Includes fallback logic for model availability and rate limits.
    """
    print(f"🌍 [Maps] Processing: {prompt}")
    
    check_limits()
    
    # Configure tools
    tools = [types.Tool(google_maps=types.GoogleMaps())]
    
    # Configure tool config (location context)
    tool_config = None
    if lat is not None and lng is not None:
        tool_config = types.ToolConfig(
            retrieval_config=types.RetrievalConfig(
                lat_lng=types.LatLng(latitude=lat, longitude=lng)
            )
        )
    
    last_error = None
    for model_name in MODEL_FALLBACKS:
        try:
            print(f"📡 [Maps] Attempting with model: {model_name}")
            
            # Simulate usage check
            if usage_stats["rpm"] >= 15:
                print(f"⚠️ [Maps] RPM limit reached for {model_name}, trying next fallback...")
                continue

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=tools,
                    tool_config=tool_config,
                ),
            )
            
            # Update stats
            usage_stats["rpm"] += 1
            # usage_stats["tpm"] += len(prompt) # rough estimate
            
            grounding_data = None
            widget_token = None
            sources = []
            
            if response.candidates and response.candidates[0].grounding_metadata:
                gm = response.candidates[0].grounding_metadata
                # Handle different attribute names in different versions of the SDK
                widget_token = getattr(gm, 'google_maps_widget_context_token', None)
                
                if gm.grounding_chunks:
                    for chunk in gm.grounding_chunks:
                        if chunk.maps:
                            sources.append({
                                "title": chunk.maps.title,
                                "uri": chunk.maps.uri,
                                "placeId": getattr(chunk.maps, 'place_id', None)
                            })
            
            # Generate the HTML visualization
            html_file = generate_maps_html(
                prompt=prompt,
                response_text=response.text,
                widget_token=widget_token,
                sources=sources,
                lat=lat or 34.0522,
                lng=lng or -118.2437,
                zoom=zoom,
                task=task,
                route_points={"from": route_from, "to": route_to} if route_from and route_to else None
            )
            
            # Open in browser
            abs_path = os.path.abspath(html_file)
            webbrowser.open(f"file:///{abs_path}")
            
            return f"Grounding result ({model_name}): {response.text}\n\nInteractive map opened: {html_file}"

        except Exception as e:
            last_error = e
            print(f"❌ [Maps] Model {model_name} failed: {e}")
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"⏳ [Maps] Rate limit or quota error on {model_name}. Switching...")
                continue
            elif "not available" in str(e).lower() or "not found" in str(e).lower():
                continue
            else:
                # Other errors might be fatal, but let's try fallback anyway
                continue
    
    return f"Maps grounding failed after all fallbacks. Last error: {last_error}"
def generate_maps_html(prompt, response_text, widget_token, sources, lat, lng, zoom, task, route_points):
    """Generates a premium, animated Google Maps visualization."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"map_viz_{timestamp}.html"
    
    # Prepare sources JSON
    sources_json = json.dumps(sources)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Shadow AI - Interactive Maps Grounding</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        :root {{
            --bg: #0a0a0c;
            --panel: rgba(20, 20, 25, 0.85);
            --accent: #3b82f6;
            --text: #e2e8f0;
            --text-dim: #94a3b8;
        }}

        body, html {{
            margin: 0; padding: 0;
            height: 100%; width: 100%;
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            overflow: hidden;
        }}

        #map-container {{
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: 1;
        }}

        #ui-overlay {{
            position: absolute;
            top: 20px; left: 20px;
            width: 400px;
            max-height: calc(100% - 40px);
            background: var(--panel);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            z-index: 10;
            padding: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
            animation: slideIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        @keyframes slideIn {{
            from {{ transform: translateX(-100%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}

        h1 {{ margin: 0 0 8px 0; font-size: 1.2rem; color: var(--accent); font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }}
        .prompt {{ font-size: 0.9rem; color: var(--text-dim); margin-bottom: 16px; font-style: italic; }}
        .response {{ font-size: 1rem; line-height: 1.6; margin-bottom: 20px; color: var(--text); overflow-y: auto; }}
        
        .sources {{
            border-top: 1px solid rgba(255,255,255,0.1);
            padding-top: 16px;
        }}
        .source-item {{
            background: rgba(255,255,255,0.05);
            padding: 8px 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            transition: all 0.2s;
            cursor: pointer;
            text-decoration: none;
            color: var(--text);
        }}
        .source-item:hover {{ background: rgba(59, 130, 246, 0.2); transform: translateX(5px); }}
        .source-icon {{ margin-right: 10px; opacity: 0.7; }}

        #widget-container {{
            margin-top: 20px;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .GMP-attribution {{
            font-family: Roboto, Sans-Serif;
            font-style: normal;
            font-weight: 400;
            font-size: 0.75rem;
            color: var(--text-dim);
            margin-top: 10px;
            text-align: right;
        }}

        /* Maps Styles */
        .gm-style-iw {{ background: var(--panel) !important; color: var(--text) !important; border-radius: 8px !important; }}
        .gm-style-iw-d {{ overflow: auto !important; }}
        .gm-style-iw-tc::after {{ background: var(--panel) !important; }}
    </style>
    
    <!-- Load Google Maps JS API -->
    <script src="https://maps.googleapis.com/maps/api/js?key={API_KEY}&libraries=places,geometry,drawing"></script>
</head>
<body>

    <div id="map-container"></div>

    <div id="ui-overlay">
        <h1>Grounding with Google Maps</h1>
        <div class="prompt">"{prompt}"</div>
        <div class="response">{response_text}</div>
        
        <div class="sources" id="source-list">
            <!-- Sources will be injected here -->
        </div>

        {f'<div id="widget-container"><gmp-place-contextual context-token="{widget_token}"></gmp-place-contextual></div>' if widget_token else ''}
        
        <div class="GMP-attribution" translate="no">Powered by Google Maps</div>
    </div>

    <script>
        let map;
        let markers = [];
        let directionsService;
        let directionsRenderer;

        const sources = {sources_json};
        const initialLat = {lat};
        const initialLng = {lng};
        const initialZoom = {zoom};
        const task = "{task}";
        const routePoints = {json.dumps(route_points) if route_points else "null"};

        function initMap() {{
            const center = {{ lat: initialLat, lng: initialLng }};
            
            map = new google.maps.Map(document.getElementById("map-container"), {{
                zoom: initialZoom,
                center: center,
                mapId: "6a67f0f6c2f90123", // Custom styled map ID (optional)
                disableDefaultUI: false,
                styles: [
                    {{ "elementType": "geometry", "stylers": [{{ "color": "#212121" }}] }},
                    {{ "elementType": "labels.icon", "stylers": [{{ "visibility": "off" }}] }},
                    {{ "elementType": "labels.text.fill", "stylers": [{{ "color": "#757575" }}] }},
                    {{ "elementType": "labels.text.stroke", "stylers": [{{ "color": "#212121" }}] }},
                    {{ "featureType": "administrative", "elementType": "geometry", "stylers": [{{ "color": "#757575" }}] }},
                    {{ "featureType": "poi", "elementType": "geometry", "stylers": [{{ "color": "#181818" }}] }},
                    {{ "featureType": "road", "elementType": "geometry.fill", "stylers": [{{ "color": "#2c2c2c" }}] }},
                    {{ "featureType": "water", "elementType": "geometry", "stylers": [{{ "color": "#000000" }}] }}
                ]
            }});

            directionsService = new google.maps.DirectionsService();
            directionsRenderer = new google.maps.DirectionsRenderer({{
                map: map,
                polylineOptions: {{
                    strokeColor: "#3b82f6",
                    strokeOpacity: 0.8,
                    strokeWeight: 6
                }}
            }});

            renderSources();
            executeTask();
        }}

        function renderSources() {{
            const list = document.getElementById("source-list");
            sources.forEach(src => {{
                const item = document.createElement("a");
                item.className = "source-item";
                item.href = src.uri;
                item.target = "_blank";
                item.innerHTML = `
                    <span class="source-icon">📍</span>
                    <span>${{src.title}}</span>
                `;
                list.appendChild(item);

                // Add Marker if we have placeId or title
                if (src.placeId) {{
                    const service = new google.maps.places.PlacesService(map);
                    service.getDetails({{ placeId: src.placeId }}, (place, status) => {{
                        if (status === google.maps.places.PlacesServiceStatus.OK) {{
                            addMarker(place.geometry.location, src.title);
                        }}
                    }});
                }}
            }});
        }}

        function addMarker(location, title) {{
            const marker = new google.maps.Marker({{
                position: location,
                map: map,
                title: title,
                animation: google.maps.Animation.DROP,
                icon: {{
                    path: google.maps.SymbolPath.CIRCLE,
                    fillColor: "#3b82f6",
                    fillOpacity: 1,
                    strokeWeight: 2,
                    strokeColor: "#ffffff",
                    scale: 10
                }}
            }});
            markers.push(marker);
            
            const infoWindow = new google.maps.InfoWindow({{
                content: `<div style="color:#000; padding:10px;"><strong>${{title}}</strong></div>`
            }});
            
            marker.addListener("click", () => {{
                infoWindow.open(map, marker);
            }});
        }}

        function executeTask() {{
            if (task === "route" && routePoints) {{
                calculateAndDisplayRoute(routePoints.from, routePoints.to);
            }} else if (task === "zoom") {{
                // Smooth zoom animation
                setTimeout(() => {{
                    map.setZoom(initialZoom + 3);
                    map.panTo({{ lat: initialLat, lng: initialLng }});
                }}, 2000);
            }}
        }}

        function calculateAndDisplayRoute(origin, destination) {{
            directionsService.route(
                {{
                    origin: origin,
                    destination: destination,
                    travelMode: google.maps.TravelMode.DRIVING
                }},
                (response, status) => {{
                    if (status === "OK") {{
                        directionsRenderer.setDirections(response);
                        // Animation: slowly pan through the route
                        animateRoute(response.routes[0].overview_path);
                    }}
                }}
            );
        }}

        function animateRoute(path) {{
            let i = 0;
            const interval = setInterval(() => {{
                if (i >= path.length) {{
                    clearInterval(interval);
                    return;
                }}
                map.panTo(path[i]);
                i += Math.floor(path.length / 20); // Move in 20 steps
            }}, 500);
        }}

        window.onload = initMap;
    </script>
</body>
</html>
    """
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return filename

if __name__ == "__main__":
    # Test
    res = run_google_maps_tool("What are the best Italian restaurants within a 15-minute walk from LA City Hall?", lat=34.0541, lng=-118.2430)
    print(res)
