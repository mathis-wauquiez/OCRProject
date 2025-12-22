#!/usr/bin/env python3
"""
Compact test suite for SVG parser v2
Usage: python test_v2.py <input.svg>
"""

import sys
import numpy as np
from ..svg import SVG, translation, rotation, scaling, shear, mirror_x, mirror_y


def test_load_save(filepath: str):
    """Test load and save"""
    print(f"Loading: {filepath}")
    svg = SVG.load(filepath)
    print(f"  {svg}")
    print(f"  Contours: {sum(len(p.contours) for p in svg.paths)}")
    print(f"  Curves: {sum(len(c) for p in svg.paths for c in p.contours)}")
    
    output = filepath.replace('.svg', '_copy.svg')
    svg.save(output)
    print(f"  Saved: {output}")
    return svg


def test_transformations(filepath: str):
    """Test all transformations"""
    
    # Get center from bounding box for centered transformations
    svg_temp = SVG.load(filepath)
    bbox = svg_temp._compute_bbox()
    center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
    
    transforms = {
        'translate': translation(100, 50),
        'scale': scaling(1.5),
        'rotate': rotation(np.radians(45)),
        'rotate_center': rotation(np.radians(45), center=center),
        'shear': shear(0.3, 0.2),
        'mirror_x': mirror_x(),
        'mirror_y': mirror_y(),
        'combined': translation(50, 50) @ rotation(np.radians(30)) @ scaling(0.8)
    }
    
    print(f"\nApplying transformations:")
    for name, H in transforms.items():
        svg = SVG.load(filepath)
        svg.apply_homography(H)
        output = filepath.replace('.svg', f'_{name}.svg')
        svg.save(output)
        print(f"  ✓ {name:15s} → {output}")


def test_properties(filepath: str):
    """Test mathematical properties"""
    svg = SVG.load(filepath)
    
    if not svg.paths or not svg.paths[0].contours:
        print("No curves to test")
        return
    
    original = svg.paths[0].contours[0].curves[0, 0].copy()
    
    # Identity
    svg_copy = SVG.load(filepath)
    svg_copy.apply_homography(np.eye(3))
    transformed = svg_copy.paths[0].contours[0].curves[0, 0]
    dist = np.linalg.norm(original - transformed)
    print(f"\nIdentity: distance = {dist:.2e} {'✓' if dist < 1e-10 else '✗'}")
    
    # Inverse
    H = translation(50, 30) @ rotation(np.radians(45)) @ scaling(2)
    H_inv = np.linalg.inv(H)
    svg_copy = SVG.load(filepath)
    svg_copy.apply_homography(H)
    svg_copy.apply_homography(H_inv)
    transformed = svg_copy.paths[0].contours[0].curves[0, 0]
    dist = np.linalg.norm(original - transformed)
    print(f"Inverse: distance = {dist:.2e} {'✓' if dist < 1e-10 else '✗'}")
    
    # Composition
    H1, H2 = translation(10, 20), rotation(np.radians(30))
    svg_sep = SVG.load(filepath)
    svg_sep.apply_homography(H1)
    svg_sep.apply_homography(H2)
    svg_comb = SVG.load(filepath)
    svg_comb.apply_homography(H2 @ H1)
    p_sep = svg_sep.paths[0].contours[0].curves[0, 0]
    p_comb = svg_comb.paths[0].contours[0].curves[0, 0]
    dist = np.linalg.norm(p_sep - p_comb)
    print(f"Composition: distance = {dist:.2e} {'✓' if dist < 1e-10 else '✗'}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_v2.py <input.svg>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print("="*70)
    print("SVG Parser v2 - Test Suite")
    print("="*70)
    
    test_load_save(filepath)
    test_transformations(filepath)
    test_properties(filepath)
    
    print("\n" + "="*70)
    print("Tests complete!")
    print("="*70)


if __name__ == "__main__":
    main()