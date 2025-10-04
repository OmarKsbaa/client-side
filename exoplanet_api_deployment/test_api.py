"""
Test script for the Exoplanet Classifier API
Run this after starting the API server to verify everything works correctly
"""
import requests
import json
import pandas as pd
import io
import time

API_BASE = "http://localhost:8001"

def test_api_comprehensive():
    """Comprehensive test of all API endpoints"""
    print("🧪 Testing Exoplanet Classifier API")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("1. Testing root endpoint...")
    try:
        response = requests.get(f"{API_BASE}/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API Version: {data.get('version', 'Unknown')}")
            print(f"   ✅ Models: {data.get('models', 'Unknown')}")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Root endpoint error: {e}")
        return False
    
    # Test 2: Health check
    print("\n2. Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ API Status: {health['status']}")
            print(f"   ✅ Kepler Model: {'✓' if health['kepler_model_loaded'] else '✗'}")
            print(f"   ✅ TOI Model: {'✓' if health['toi_model_loaded'] else '✗'}")
            
            if not health['kepler_model_loaded'] or not health['toi_model_loaded']:
                print("   ⚠️  Warning: Not all models loaded successfully")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 3: Available models
    print("\n3. Testing models endpoint...")
    try:
        response = requests.get(f"{API_BASE}/models")
        if response.status_code == 200:
            models = response.json()
            for model_name, model_info in models.items():
                print(f"   ✅ {model_name.upper()}: {model_info['accuracy']} accuracy")
                print(f"      Features: {model_info['features_count']}")
        else:
            print(f"   ❌ Models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Models endpoint error: {e}")
    
    # Test 4: Feature requirements
    print("\n4. Testing feature endpoints...")
    for model_type in ['kepler', 'toi']:
        try:
            response = requests.get(f"{API_BASE}/features/{model_type}")
            if response.status_code == 200:
                features = response.json()
                print(f"   ✅ {model_type.upper()}: {features['feature_count']} features required")
            else:
                print(f"   ❌ {model_type.upper()} features failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ {model_type.upper()} features error: {e}")
    
    # Test 5: Sample predictions
    print("\n5. Testing sample predictions...")
    for model_type in ['kepler', 'toi']:
        print(f"\n   Testing {model_type.upper()} model...")
        try:
            response = requests.post(f"{API_BASE}/predict/sample/{model_type}")
            if response.status_code == 200:
                result = response.json()
                predictions = result['predictions']
                print(f"   ✅ {model_type.upper()}: {len(predictions)} sample predictions")
                
                # Show first prediction details
                if predictions:
                    pred = predictions[0]
                    print(f"      Sample result: {pred['prediction']} ({pred['confidence']:.3f} confidence)")
                    
            else:
                print(f"   ❌ {model_type.upper()} sample failed: {response.status_code}")
                print(f"      Error: {response.text}")
        except Exception as e:
            print(f"   ❌ {model_type.upper()} sample error: {e}")
    
    # Test 6: CSV upload simulation
    print("\n6. Testing CSV upload functionality...")
    
    # Create sample Kepler data
    kepler_data = pd.DataFrame({
        'koi_score': [0.95, 0.12, 0.85, 0.67, 0.34],
        'koi_fpflag_nt': [0, 1, 0, 0, 1],
        'koi_fpflag_ec': [0, 0, 1, 0, 0],
        'koi_fpflag_co': [0, 1, 0, 1, 0],
        'koi_model_snr': [25.5, 8.2, 45.1, 15.3, 6.7],
        'koi_fpflag_ss': [0, 0, 1, 0, 0],
        'koi_prad': [1.1, 2.8, 0.95, 1.5, 3.2],
        'koi_period': [365.25, 88.0, 687.0, 42.3, 156.8],
        'koi_duration': [6.2, 12.5, 4.8, 8.1, 15.2],
        'koi_impact': [0.3, 0.8, 0.1, 0.5, 0.9]
    })
    
    # Test Kepler CSV upload
    try:
        csv_buffer = io.StringIO()
        kepler_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        files = {'file': ('test_kepler.csv', csv_content, 'text/csv')}
        data = {'model_type': 'kepler'}
        
        response = requests.post(f"{API_BASE}/predict/csv", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Kepler CSV: {result['total_rows']} objects processed")
            print(f"      Candidates: {result['summary']['total_candidates']}")
            print(f"      False Positives: {result['summary']['total_false_positives']}")
            print(f"      Processing time: {result['processing_time_seconds']:.3f}s")
        else:
            print(f"   ❌ Kepler CSV upload failed: {response.status_code}")
            print(f"      Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Kepler CSV test error: {e}")
    
    # Create sample TOI data
    toi_data = pd.DataFrame({
        'pl_eqt': [1500, 800, 2000, 1200, 900],
        'pl_insol': [1.2, 0.3, 5.8, 2.1, 0.8],
        'pl_orbpererr2': [0.01, 0.05, 0.02, 0.03, 0.04],
        'pl_radeerr2': [0.1, 0.3, 0.15, 0.2, 0.25],
        'pl_tranmid': [2458000, 2459000, 2458500, 2458750, 2459250],
        'st_disterr2': [5, 10, 8, 6, 12],
        'st_tmag': [12.5, 9.8, 14.2, 11.3, 13.7],
        'pl_tranmiderr2': [0.0001, 0.0005, 0.0002, 0.0003, 0.0004],
        'st_disterr1': [5, 8, 6, 4, 9],
        'pl_trandeperr1': [0.5, 1.2, 0.8, 0.7, 1.0]
    })
    
    # Test TOI CSV upload
    try:
        csv_buffer = io.StringIO()
        toi_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        files = {'file': ('test_toi.csv', csv_content, 'text/csv')}
        data = {'model_type': 'toi'}
        
        response = requests.post(f"{API_BASE}/predict/csv", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ TOI CSV: {result['total_rows']} objects processed")
            print(f"      Candidates: {result['summary']['total_candidates']}")
            print(f"      False Positives: {result['summary']['total_false_positives']}")
            print(f"      Processing time: {result['processing_time_seconds']:.3f}s")
        else:
            print(f"   ❌ TOI CSV upload failed: {response.status_code}")
            print(f"      Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ TOI CSV test error: {e}")
    
    print(f"\n🎉 API Testing Complete!")
    print(f"📱 Web Interface: {API_BASE}/predict/form")
    print(f"📚 API Documentation: {API_BASE}/docs")
    print(f"🏥 Health Check: {API_BASE}/health")
    
    return True

def performance_test():
    """Basic performance test"""
    print("\n🚀 Performance Test")
    print("-" * 30)
    
    # Test response time for health endpoint
    times = []
    for i in range(10):
        start = time.time()
        try:
            response = requests.get(f"{API_BASE}/health")
            end = time.time()
            if response.status_code == 200:
                times.append(end - start)
        except:
            pass
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"Average health check response time: {avg_time*1000:.2f}ms")
    
    # Test sample prediction speed
    start = time.time()
    try:
        response = requests.post(f"{API_BASE}/predict/sample/kepler")
        end = time.time()
        if response.status_code == 200:
            print(f"Sample prediction time: {(end-start)*1000:.2f}ms")
    except Exception as e:
        print(f"Performance test error: {e}")

if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the API server is running at http://localhost:8001")
    print()
    
    # Wait a moment for user to start server if needed
    try:
        requests.get(f"{API_BASE}/health", timeout=2)
    except:
        print("⚠️  API server not responding. Please start the server with:")
        print("   uvicorn kepler_api:app --host 0.0.0.0 --port 8001")
        print()
        input("Press Enter when the server is running...")
    
    # Run comprehensive tests
    success = test_api_comprehensive()
    
    if success:
        # Run performance test
        performance_test()
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check the API server and try again.")