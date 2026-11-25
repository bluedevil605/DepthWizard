import { createProceduralTerrainCanvas } from './depthMapGenerator';

export interface SampleImage {
  id: string;
  name: string;
  category: string;
  description: string;
  dataUrl: string;
  preset: 'flood' | 'landslide' | 'collapse' | 'mountain';
  depthMetrics: {
    maxElevation: string;
    waterVolume: string;
    hazardIndex: string;
    confidence: string;
  };
}

// Generate base64 data URLs for immediate preset loading without external network calls
export function getSampleDisasterImages(): SampleImage[] {
  return [
    {
      id: 'flood-zone',
      name: 'Kerala River Basin Flood',
      category: 'Hydrological Disaster',
      description: 'Single aerial drone view of inundated river basin with high-resolution flood plain elevation mapping.',
      dataUrl: createProceduralTerrainCanvas('flood', 512).toDataURL('image/png'),
      preset: 'flood',
      depthMetrics: {
        maxElevation: '+14.2m',
        waterVolume: '4.8M m³',
        hazardIndex: 'HIGH (Level 4)',
        confidence: '96.4%'
      }
    },
    {
      id: 'landslide-wayanad',
      name: 'Wayanad Hillside Slip',
      category: 'Geological Slope Collapse',
      description: 'Monsoon hillside landslide with steep escarpment depth profile and mud debris tracking.',
      dataUrl: createProceduralTerrainCanvas('landslide', 512).toDataURL('image/png'),
      preset: 'landslide',
      depthMetrics: {
        maxElevation: '+185.0m',
        waterVolume: '0.9M m³ debris',
        hazardIndex: 'CRITICAL (Level 5)',
        confidence: '94.8%'
      }
    },
    {
      id: 'turkey-earthquake',
      name: 'Urban Building Collapse',
      category: 'Structural Collapse',
      description: 'Post-earthquake rubble field depth estimation for urban search & rescue (USAR) void detection.',
      dataUrl: createProceduralTerrainCanvas('collapse', 512).toDataURL('image/png'),
      preset: 'collapse',
      depthMetrics: {
        maxElevation: '+8.6m rubble',
        waterVolume: '12 void pockets',
        hazardIndex: 'EXTREME (Level 5)',
        confidence: '92.1%'
      }
    },
    {
      id: 'himalayan-valley',
      name: 'Himalayan Flash Flood Sector',
      category: 'Disaster Relief Pathfinding',
      description: 'Glacial outburst terrain profile mapping for autonomous UAV logistics flight path planning.',
      dataUrl: createProceduralTerrainCanvas('mountain', 512).toDataURL('image/png'),
      preset: 'mountain',
      depthMetrics: {
        maxElevation: '+620.0m',
        waterVolume: '1.2M m³ flow',
        hazardIndex: 'MODERATE (Level 3)',
        confidence: '97.9%'
      }
    }
  ];
}
