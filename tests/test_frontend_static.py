import re
from pathlib import Path

def test_frontend_html_and_ids():
    index_path = Path("app/frontend/index.html")
    assert index_path.exists(), "index.html does not exist"
    
    content = index_path.read_text(encoding="utf-8")
    
    # 1. Check title and main structure
    assert "<title>Vehicle Damage Scanner" in content
    assert 'id="view-upload"' in content
    assert 'id="view-analyzing"' in content
    assert 'id="view-results"' in content
    
    # 2. Check upload screen elements
    assert 'id="dropzone"' in content
    assert 'id="file-input"' in content
    assert 'id="dropzone-empty-content"' in content
    assert 'id="dropzone-preview-overlay"' in content
    assert 'id="preview-image"' in content
    assert 'id="preview-filename"' in content
    assert 'id="preview-filemeta"' in content
    assert 'id="btn-analyze"' in content
    
    # 3. Check that no unsupported marketing claims exist
    assert "100% Private" not in content
    assert "5-second scan" not in content
    assert "Calibrated YOLOv8" not in content
    assert "Stress-Free Assessment" not in content
    
    # 4. Check three photography tips
    assert "Good lighting" in content
    assert "Wide angle" in content
    assert "Clean surface" in content
    
    # 5. Extract all getElementById calls and verify matching HTML IDs
    html_ids = set(re.findall(r'id=["\']([^"\']+)["\']', content))
    js_ids = set(re.findall(r'getElementById\(["\']([^"\']+)["\']\)', content))
    
    missing_ids = js_ids - html_ids
    assert not missing_ids, f"Elements accessed by getElementById missing in HTML: {missing_ids}"
    
    # 6. Verify initial disabled state of analyze button
    assert 'id="btn-analyze"' in content
    assert 'disabled' in content[content.find('id="btn-analyze"') - 30:content.find('id="btn-analyze"') + 60]
