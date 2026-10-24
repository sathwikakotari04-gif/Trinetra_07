let hazardMarkers = [];
let liveBusMarker = null;
let deviceGpsMarker = null;
let accuracyCircle = null;

// Global tracking flags
let autoFollow = true;
let highAccuracy = true;
let watchId = null;
let currentLat = null;
let currentLng = null;

function getSeverityColor(severity) {
    const s = (severity || "").toUpperCase();
    if (s === "CRITICAL" || s === "SEVERE") return "#ff3333";
    if (s === "MODERATE") return "#ffcc00";
    return "#33cc33";
}

// Custom pulsing bus marker for real-time detected GPS
function getBusIcon() {
    return L.divIcon({
        className: 'live-bus-pin',
        html: `
            <div style="position:relative; width:26px; height:26px; display:flex; align-items:center; justify-content:center;">
                <div style="position:absolute; width:26px; height:26px; border-radius:50%; background:#00E5FF; opacity:0.4; animation:pulse 1.5s infinite;"></div>
                <div style="width:14px; height:14px; border-radius:50%; background:#00E5FF; border:2.5px solid #ffffff; box-shadow:0 0 10px #00E5FF;"></div>
            </div>
        `,
        iconSize: [26, 26],
        iconAnchor: [13, 13]
    });
}

async function updateLiveMap(mapInstance) {
    if (!mapInstance) return;

    try {
        const response = await fetch('/api/logs');
        if (!response.ok) return;

        const data = await response.json();
        const logs = data.logs || [];

        // Clear existing detection markers
        hazardMarkers.forEach(marker => mapInstance.removeLayer(marker));
        hazardMarkers = [];

        let latestCoord = null;

        // Plot detected potholes and hazards from logs
        logs.forEach(item => {
            const lat = parseFloat(item.Latitude);
            const lon = parseFloat(item.Longitude);

            if (!isNaN(lat) && !isNaN(lon)) {
                latestCoord = [lat, lon];
                const color = getSeverityColor(item.Severity);

                const marker = L.circleMarker([lat, lon], {
                    radius: 8,
                    fillColor: color,
                    color: "#ffffff",
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.85
                });

                const popupContent = `
                    <div style="font-family: sans-serif; font-size: 13px; color: #111;">
                        <b>Bus:</b> AP 123<br>
                        <b>Hazard:</b> ${item.Hazard_Type}<br>
                        <b>Severity:</b> <span style="color:${color}; font-weight:bold;">${item.Severity}</span><br>
                        <b>Confidence:</b> ${item.Confidence}<br>
                        <b>Time:</b> ${item.Timestamp}
                    </div>
                `;

                marker.bindPopup(popupContent);
                marker.addTo(mapInstance);
                hazardMarkers.push(marker);
            }
        });

        // Update live GPS Bus Marker to the latest detection coordinate
        if (latestCoord) {
            if (!liveBusMarker) {
                liveBusMarker = L.marker(latestCoord, { icon: getBusIcon(), zIndexOffset: 1000 }).addTo(mapInstance);
                liveBusMarker.bindPopup(`
                    <div style="font-family: sans-serif; font-size: 13px; color: #111;">
                        <b style="color:#00838F;">BUS AP 123 (Active Unit)</b><br>
                        Live Telemetry Tracking Active
                    </div>
                `);
            } else {
                liveBusMarker.setLatLng(latestCoord);
            }

            // Sync the on-screen live GPS text overlay
            const gpsLabel = document.getElementById('gps-coords');
            if (gpsLabel && !currentLat) {
                gpsLabel.textContent = `GPS: ${latestCoord[0].toFixed(4)}° N, ${latestCoord[1].toFixed(4)}° E`;
            }

            if (autoFollow && !currentLat) {
                mapInstance.setView(latestCoord, mapInstance.getZoom() || 16);
            }
        }
    } catch (err) {
        console.error("Failed to update map markers:", err);
    }
}

// Device Hardware GPS Tracking
function startHardwareGPS(mapInstance) {
    if (watchId !== null) navigator.geolocation.clearWatch(watchId);

    if (navigator.geolocation) {
        watchId = navigator.geolocation.watchPosition(
            (pos) => {
                currentLat = pos.coords.latitude;
                currentLng = pos.coords.longitude;
                const accuracy = pos.coords.accuracy;

                const gpsLabel = document.getElementById('gps-coords');
                if (gpsLabel) {
                    gpsLabel.textContent = `GPS: ${currentLat.toFixed(4)}° N, ${currentLng.toFixed(4)}° E`;
                }

                if (mapInstance) {
                    if (!deviceGpsMarker) {
                        const pulseIcon = L.divIcon({
                            className: 'custom-hardware-pin',
                            html: `<div style="width:14px; height:14px; background:#10B981; border:2.5px solid #fff; border-radius:50%; box-shadow:0 0 10px #10B981;"></div>`,
                            iconSize: [14, 14],
                            iconAnchor: [7, 7]
                        });
                        deviceGpsMarker = L.marker([currentLat, currentLng], { icon: pulseIcon }).addTo(mapInstance);
                    } else {
                        deviceGpsMarker.setLatLng([currentLat, currentLng]);
                    }

                    if (autoFollow) {
                        mapInstance.setView([currentLat, currentLng], mapInstance.getZoom() || 16);
                    }
                }
            },
            (err) => console.warn('Geolocation error:', err.message),
            { enableHighAccuracy: highAccuracy, maximumAge: 2000, timeout: 5000 }
        );
    }
}