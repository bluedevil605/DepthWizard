import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, ArrowRight, ShieldCheck, Play } from 'lucide-react';
import { playScanPulseSound } from '../utils/soundEffects';

interface ScanningChamberProps {
  onUpload: (file: File) => void;
}

export const ScanningChamber: React.FC<ScanningChamberProps> = ({ onUpload }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const sampleImages = [
    { id: 'sample-01', name: 'Mount_St_Helens.tif', resolution: '10m/px', type: 'GeoTIFF' },
    { id: 'sample-02', name: 'Alpine_Drone_Survey.jpg', resolution: '4K', type: 'Optical' }
  ];

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!isDragOver) {
      setIsDragOver(true);
      playScanPulseSound();
    }
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onUpload(files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onUpload(files[0]);
    }
  };

  return (
    <section id="upload-chamber" className="relative py-24 bg-[#211E1B]/75 backdrop-blur-md border-t border-[#5A564C]/50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        {/* Section Header */}
        <div className="text-center max-w-2xl mx-auto mb-12 font-mono">
          <div className="text-xs text-[#8B8478] tracking-[0.2em] uppercase mb-2">
            [ FIELD RECONNAISSANCE CHAMBER ]
          </div>
          <h2 className="font-heading text-3xl sm:text-4xl font-bold text-[#D9D2C2] uppercase tracking-tight">
            INGEST FIELD OPTICAL
          </h2>
          <p className="mt-2 text-xs text-[#B7AF9F] leading-relaxed">
            Drag handheld, drone, or satellite imagery (PNG, JPG, TIFF) to initiate DepthWizard 3D reconstruction.
          </p>
        </div>

        {/* Wireframe Scanning Chamber Upload Zone */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Chamber Box (7 cols) */}
          <div className="lg:col-span-7">
            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.tif,.tiff"
              onChange={handleFileChange}
              className="hidden"
            />

            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`relative py-16 px-8 border-2 border-dashed cursor-pointer transition-all duration-300 flex flex-col items-center justify-center text-center font-mono ${
                isDragOver
                  ? 'border-[#E0433C] bg-[#5C1315]/40 backdrop-blur-md shadow-[0_0_30px_rgba(224,67,60,0.25)] scale-[1.01]'
                  : 'border-[#5A564C] bg-[#1A1816]/80 backdrop-blur-md hover:border-[#8B8478] hover:bg-[#211E1B]/90'
              }`}
            >
              {isDragOver && (
                <div className="absolute inset-0 border border-[#E0433C] animate-ping opacity-30 pointer-events-none" />
              )}

              <div className="w-14 h-14 border border-[#5A564C] bg-[#211E1B] flex items-center justify-center text-[#D9D2C2] mb-6">
                <Upload className="w-6 h-6 text-[#E0433C]" />
              </div>

              <div className="text-sm font-bold tracking-widest text-[#D9D2C2] uppercase mb-2">
                DROP IMAGE OR GEOTIFF
              </div>
              <div className="text-xs text-[#8B8478] tracking-wider mb-6">
                SUPPORTED: PNG, JPG, TIFF, GEOTIFF
              </div>

              <div className="text-[11px] text-[#B7AF9F] border-t border-[#5A564C] pt-4 flex items-center gap-4 uppercase tracking-widest">
                <span>SCAN</span>
                <ArrowRight className="w-3.5 h-3.5 text-[#8B8478]" />
                <span>DEPTH</span>
                <ArrowRight className="w-3.5 h-3.5 text-[#8B8478]" />
                <span>DSM</span>
              </div>
            </div>
          </div>

          {/* Sample Preset Field Footage (5 cols) */}
          <div className="lg:col-span-5 font-mono space-y-4">
            <div className="text-xs text-[#8B8478] tracking-wider uppercase flex items-center justify-between border-b border-[#5A564C] pb-2">
              <span>PRESET FIELD IMAGES (DEMO)</span>
              <ImageIcon className="w-3.5 h-3.5 text-[#B7AF9F]" />
            </div>

            <div className="space-y-3">
              {sampleImages.map((sample) => (
                <div
                  key={sample.id}
                  className="w-full p-4 mono-panel border border-[#5A564C] opacity-50 cursor-not-allowed transition-all text-left group flex items-center justify-between"
                  title="Use direct file upload above"
                >
                  <div>
                    <div className="text-xs font-bold text-[#D9D2C2] transition-colors">
                      {sample.name}
                    </div>
                    <div className="text-[10px] text-[#8B8478] mt-1 flex items-center gap-3">
                      <span>{sample.type}</span>
                      <span>•</span>
                      <span>{sample.resolution}</span>
                    </div>
                  </div>
                  <Play className="w-4 h-4 text-[#8B8478] transition-colors" />
                </div>
              ))}
            </div>

            <div className="p-4 bg-[#1A1816] border border-[#5A564C] text-[11px] text-[#8B8478] space-y-1.5 mt-6">
              <div className="flex items-center gap-2 text-[#B7AF9F] font-semibold">
                <ShieldCheck className="w-3.5 h-3.5 text-[#E0433C]" />
              <span>DEPTH PROCESSING ENGINE</span>
              </div>
              <p>
                Powered by Depth Anything V2. Depth estimation and GLB generation run on the DepthWizard backend.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
