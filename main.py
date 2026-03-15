import os
import ee
import warnings
import torch
import torch.nn as nn
import numpy as np
import rasterio
import io
import requests
import base64
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from geopy.geocoders import Nominatim
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
# Enable CORS with support for credentials and explicit options handling
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# --- INITIALIZE EARTH ENGINE (Cloud-Ready) ---
try:
    ee_project = os.getenv('EE_PROJECT_ID', 'lulc-470905')
    service_account = os.getenv('EE_SERVICE_ACCOUNT')
    private_key = os.getenv('EE_PRIVATE_KEY')

    if service_account and private_key:
        # PRODUCTION: Service Account Auth (Render)
        formatted_key = private_key.replace('\\n', '\n')
        credentials = ee.ServiceAccountCredentials(service_account, key_data=formatted_key)
        ee.Initialize(credentials, project=ee_project)
        print(f"✅ EE Initialized via Service Account: {service_account}")
    else:
        # LOCAL: Default User Auth
        ee.Initialize(project=ee_project)
        print(f"✅ EE Initialized via Local Auth (Project: {ee_project})")
except Exception as e:
    print(f"❌ EE Initialization Error: {e}")

# --- U-NET MODEL ARCHITECTURE ---
device = torch.device('cpu') 
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def cb(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        self.enc1 = cb(10, 64); self.pool = nn.MaxPool2d(2)
        self.enc2 = cb(64, 128); self.enc3 = cb(128, 256)
        self.bottleneck = cb(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2); self.dec3 = cb(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2); self.dec2 = cb(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2); self.dec1 = cb(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

model_net = UNet().to(device)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'tvm_growth_model.pth')

if os.path.exists(MODEL_PATH):
    try:
        model_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model_net.eval()
        print("✅ AI Prediction Model Loaded Successfully")
    except Exception as e:
        print(f"❌ Model Weight Error: {e}")
else:
    print(f"⚠️ Warning: Model weights not found at {MODEL_PATH}")

# --- SHARED DATA ---
DW_CLASSES = {0: "Water", 1: "Trees", 2: "Grass", 3: "Flooded Vegetation", 4: "Crops", 5: "Shrub & Scrub", 6: "Built Area", 7: "Bare Ground", 8: "Snow & Ice"}
DW_PALETTE = ['419bdf', '397d49', '88b053', '7a87c6', 'e49635', 'dfc35a', 'c4281b', 'a59b8f', 'b39fe1']

# --- LULC CHANGE ANALYSIS ---
@app.route('/api/analyze-lulc', methods=['POST', 'OPTIONS'])
def analyze_lulc():
    if request.method == 'OPTIONS': return '', 200
    try:
        data = request.json
        year_start = int(data.get('year_start', 2016))
        year_end = int(data.get('year_end', 2024))
        region = ee.Geometry(data['geojson'])
        
        dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(region)
        img_s = dw.filterDate(f'{year_start}-01-01', f'{year_start}-12-31').select('label').mode().clip(region)
        img_e = dw.filterDate(f'{year_end}-01-01', f'{year_end}-12-31').select('label').mode().clip(region)
        
        transition_img = img_s.multiply(10).add(img_e)
        area_image = ee.Image.pixelArea().addBands(transition_img.updateMask(img_s.neq(img_e)))
        stats = area_image.reduceRegion(reducer=ee.Reducer.sum().group(groupField=1, groupName='code'), geometry=region, scale=10, maxPixels=1e13)
        
        groups = stats.get('groups').getInfo()
        results = []
        total_km2 = 0
        if groups:
            for g in groups:
                area_km2 = g['sum'] / 1e6
                if area_km2 > 0.0001:
                    total_km2 += area_km2
                    from_code, to_code = int(g['code'] // 10), int(g['code'] % 10)
                    results.append({
                        'from': DW_CLASSES.get(from_code, "Unknown"), 
                        'to': DW_CLASSES.get(to_code, "Unknown"), 
                        'area_km2': round(area_km2, 4), 
                        'raw': area_km2
                    })
        
        results.sort(key=lambda x: x['raw'], reverse=True)
        return jsonify({
            'total_changed_km2': round(total_km2, 4), 
            'dominant_shift': results[0]['to'] if results else "Stable", 
            'transitions': results[:10],
            'analyzed_period': f'{year_start}-{year_end}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- AI INSIGHT GENERATION ---
@app.route('/api/generate-inference', methods=['POST', 'OPTIONS'])
def generate_inference():
    if request.method == 'OPTIONS': return '', 200
    data = request.json
    coords = data.get('coords')
    location_full = "the analyzed area"
    
    if coords:
        try:
            geolocator = Nominatim(user_agent="lulc_seminar_app", timeout=5)
            location = geolocator.reverse(f"{coords['lat']}, {coords['lon']}", language='en')
            if location:
                addr = location.raw.get('address', {})
                city = addr.get('city') or addr.get('town') or "Area"
                location_full = f"{city}, {addr.get('state', 'India')}"
        except: pass
    
    prompt = f"As a Senior Urban Planner, analyze {location_full}. Total change: {data.get('total_changed_km2')} km2, primary shift: {data.get('dominant_shift')}. Write a 100-word professional report on urban expansion."
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return jsonify({'inference': 'Urban growth detected. (API Key Missing)', 'used_model': 'LOCAL HEURISTIC', 'location': location_full})

    models_to_try = ['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemma-3-27b-it']
    client = genai.Client(api_key=api_key)

    for model_id in models_to_try:
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            return jsonify({
                'inference': response.text, 
                'used_model': model_id.upper().replace("-", " "), 
                'location': location_full
            })
        except Exception as e:
            print(f"⚠️ Model {model_id} failed: {e}")
            continue

    return jsonify({'inference': 'Urban growth detected.', 'used_model': 'LOCAL HEURISTIC', 'location': location_full})

# --- MAP LAYER & PDF THUMBNAIL ROUTES ---
@app.route('/api/get-gee-layer', methods=['POST', 'OPTIONS'])
def get_gee_layer():
    if request.method == 'OPTIONS': return '', 200
    try:
        data = request.json
        year, region = int(data['year']), ee.Geometry(data['geojson'])
        img = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate(f'{year}-01-01', f'{year}-12-31').filterBounds(region).select('label').mode().clip(region)
        map_id = img.getMapId({'min': 0, 'max': 8, 'palette': DW_PALETTE})
        return jsonify({'url': map_id['tile_fetcher'].url_format})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-lulc-thumb', methods=['POST', 'OPTIONS'])
def get_lulc_thumb():
    if request.method == 'OPTIONS': return '', 200
    try:
        data = request.json
        year, region = int(data.get('year', 2024)), ee.Geometry(data['geojson'])
        img = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate(f'{year}-01-01', f'{year}-12-31').filterBounds(region).select('label').mode().clip(region)
        # Dimensions 500 for faster PDF rendering
        return jsonify({'url': img.getThumbURL({'min': 0, 'max': 8, 'palette': DW_PALETTE, 'dimensions': 500, 'region': region, 'format': 'png'})})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-satellite-thumb', methods=['POST', 'OPTIONS'])
def get_satellite_thumb():
    if request.method == 'OPTIONS': return '', 200
    try:
        data = request.json
        year, region = int(data.get('year', 2024)), ee.Geometry(data['geojson'])
        img = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(region).filterDate(f'{year}-01-01', f'{year}-12-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)).median().clip(region)
        thumb_url = img.getThumbURL({'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.4, 'region': region, 'dimensions': 500, 'format': 'png'})
        return jsonify({'url': thumb_url})
    except Exception as e:
        return jsonify({'url': 'https://via.placeholder.com/500?text=Satellite+Unavailable'})

# --- FUTURE GROWTH PREDICTION (U-NET) ---
@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict_heatmap():
    if request.method == 'OPTIONS': return '', 200
    try:
        data = request.json
        roi = ee.Geometry(data['geojson'])
        dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate('2023-01-01', '2023-12-31').filterBounds(roi).mode()
        input_image = ee.Image.cat([dw.select('label'), dw.select('label').eq(6).fastDistanceTransform(30).sqrt().multiply(30)]).toFloat().clip(roi)
        
        url = input_image.getDownloadURL({'scale': 30, 'crs': 'EPSG:4326', 'region': roi, 'format': 'GEO_TIFF'})
        resp = requests.get(url)
        with rasterio.open(io.BytesIO(resp.content)) as src:
            img_data = src.read()
            bounds = src.bounds
            h, w = img_data.shape[1], img_data.shape[2]
            
        ph, pw = (16 - (h % 16)) % 16, (16 - (w % 16)) % 16
        padded = np.pad(img_data, ((0,0), (0, ph), (0, pw)), mode='edge')
        lulc_onehot = np.eye(9)[np.clip(padded[0].astype(int), 0, 8)]
        dist_band = np.expand_dims(np.clip(padded[1]/5000.0, 0, 1), -1)
        tensor = torch.from_numpy(np.concatenate([lulc_onehot, dist_band], -1)).permute(2,0,1).float().unsqueeze(0)
        
        with torch.no_grad():
            probs = torch.sigmoid(model_net(tensor)).squeeze().numpy()[:h, :w]
            
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask = probs > 0.2
        rgba[mask] = [255, 0, 0, 160]
        buff = io.BytesIO()
        Image.fromarray(rgba).save(buff, format="PNG")
        return jsonify({
            "url": f"data:image/png;base64,{base64.b64encode(buff.getvalue()).decode()}",
            "bounds": [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 LULC Unified Server Starting on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)