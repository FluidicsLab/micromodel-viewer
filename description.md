# Draft Microfluidic Chip Design Workflow  
---

## 1. Context: Pore Network Mimicry in Oil Shale

Oil shale reservoirs contain extremely tight porous structures with nanoscale pore throats — far below the resolution of conventional planar microfabrication.

### Key challenge  
Lateral width variation cannot capture the true pore throat distribution.

### Approach  
- Keep channel widths constant for clean fabrication  
- Encode pore size variation in the vertical dimension (channel depth)  
- Maintain structural clarity and reproducibility  

---

## 2. Core Innovation

To match MICP data:

- Use 2D Voronoi geometry  
  → to mimic disordered pore topology  
- Assign a depth value  
  → to each throat from MICP bins  
- Maintain uniform width  
  → for all throats, manufacturable via photolithography

---

## 3. Workflow Summary

| Stage                          | Method                          | Output                         |
|-------------------------------|----------------------------------|--------------------------------|
| Pore geometry synthesis       | Voronoi tessellation             | 2D abstract pore network       |
| Dimension assignment          | Length from geometry + depth map | Volume-mapped network          |
| Manufacturing format          | Constant width, variable depth   | DXF for lithography            |

---

## 4. Input: MICP-Derived Pore Size Distribution

MICP data is segmented into discrete depth bins, each including:

- A characteristic pore throat depth  
- A target volume fraction

These are used to assign depth during chip generation.

---

## 5. Synthetic Network Generation

- Generate random seed points
- Apply 2D Voronoi tessellation  
- Treat each ridge as a pore throat  
- Clip throats to design boundaries  
- Result: constant-width, variable-depth geometry

---

## 6. Depth Assignment with Volume Constraints

Each Voronoi ridge (throat) is assigned:

- Fixed width
- Measured length  
- Depth based on MICP volume bins

This ensures that the volume distribution matches MICP data.

---