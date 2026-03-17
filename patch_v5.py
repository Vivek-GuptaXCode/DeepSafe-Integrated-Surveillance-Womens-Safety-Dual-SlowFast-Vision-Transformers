import json

def apply_patch():
    with open('integrated-women-safety-system.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = "".join(cell['source'])

        # ─── PATCH 1: Improve Grad-CAM generate() with sharpening ───
        if 'class SlowFastGradCAM:' in source:
            # 1a. Sharpen the cam with power-law before upsampling
            old_cam_norm = """        cam = cam[0].detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)"""

            new_cam_norm = """        cam = cam[0].detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
            # Sharpen: power-law emphasizes peaks, suppresses low activations
            cam = cam ** 2.0
            # Re-normalize after sharpening
            cam = cam / (cam.max() + 1e-8)
        else:
            cam = np.zeros_like(cam)"""

            source = source.replace(old_cam_norm, new_cam_norm)

            # 1b. Use INTER_CUBIC for smoother upsampling and apply the
            # overlay only within the crop region (masked blend)
            old_overlay = """        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        overlay = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)"""

            new_overlay = """        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        overlay_raw = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        # Mask: only show heatmap where activation is significant (> 0.15)
        # This prevents the uniform blue tint over the entire frame
        mask = (heatmap > 0.15).astype(np.float32)
        mask_3ch = np.stack([mask]*3, axis=-1)
        overlay = (overlay_raw * mask_3ch).astype(np.uint8)"""

            source = source.replace(old_overlay, new_overlay)

        # ─── PATCH 2: Use blocks[4] instead of blocks[5] for higher spatial resolution ───
        if 'SlowFastGradCAM(model_rgb, model_flow, fusion, target_block_idx=5)' in source:
            source = source.replace(
                'SlowFastGradCAM(model_rgb, model_flow, fusion, target_block_idx=5)',
                'SlowFastGradCAM(model_rgb, model_flow, fusion, target_block_idx=4)'
            )
            source = source.replace(
                "print(f'  Backprop target: fused Fight logit from FusionClassifier (fight_index={CONFIG.get(\"fight_index\", 0)})')",
                "print(f'  Backprop target: fused Fight logit from FusionClassifier (fight_index={CONFIG.get(\"fight_index\", 0)}, block=4 for sharper maps)')"
            )

        # ─── PATCH 3: Annotate violence bounding boxes on persons ───
        if 'def annotate_frame(' in source:
            # After evidence bboxes, add violence person bounding boxes
            old_evidence_section = """    # ── Evidence bounding boxes (red) ──
    if evidence_bboxes and smoothed_violence == 'Fight':
        for (x, y, w, h) in evidence_bboxes:
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(annotated, 'EVIDENCE', (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)"""

            new_evidence_section = """    # ── Violence region bounding boxes ──
    if smoothed_violence == 'Fight':
        # Draw evidence contour bboxes if available
        if evidence_bboxes:
            for (x, y, w, h) in evidence_bboxes:
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(annotated, 'VIOLENCE ZONE', (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw thick red bounding boxes around all tracked persons
        # to indicate they are in a violence region
        for track in active_tracks:
            x1, y1, x2, y2 = track.bbox
            # Thick red border signals violence involvement
            cv2.rectangle(annotated, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 3)
            # Violence confidence label
            viol_label = 'VIOLENCE'
            (tw2, th2), _ = cv2.getTextSize(viol_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y2+2), (x1+tw2+4, y2+th2+8), (0, 0, 255), -1)
            cv2.putText(annotated, viol_label, (x1+2, y2+th2+4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)"""

            source = source.replace(old_evidence_section, new_evidence_section)

        # write it back
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source'][-1] == '\n':
            cell['source'].pop()
        else:
            cell['source'][-1] = cell['source'][-1][:-1]

    with open('integrated-women-safety-system.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

    print("Patch v5 (Grad-CAM focus + violence bboxes) applied successfully.")

if __name__ == '__main__':
    apply_patch()
