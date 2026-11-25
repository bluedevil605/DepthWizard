import React, { useEffect, useRef } from 'react';

export const WebGLFallback: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || window.innerWidth);
    let height = (canvas.height = canvas.parentElement?.clientHeight || window.innerHeight);

    const handleResize = () => {
      if (!canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener('resize', handleResize);

    let time = 0;

    const render = () => {
      time += 0.015;
      ctx.clearRect(0, 0, width, height);

      // Draw animated 2D perspective wireframe grid
      const horizon = height * 0.45;
      const lines = 24;
      const step = width / lines;

      ctx.strokeStyle = 'rgba(0, 212, 255, 0.12)';
      ctx.lineWidth = 1;

      // Perspective vertical grid lines
      for (let i = 0; i <= lines; i++) {
        const x = i * step;
        ctx.beginPath();
        ctx.moveTo(width / 2, horizon);
        ctx.lineTo((x - width / 2) * 3 + width / 2, height);
        ctx.stroke();
      }

      // Horizontal wave lines (Simulated 3D terrain profile)
      const rows = 18;
      for (let j = 0; j < rows; j++) {
        const progress = (j / rows + time * 0.05) % 1;
        const y = horizon + Math.pow(progress, 2.2) * (height - horizon);
        const wave = Math.sin(j * 0.5 + time * 2) * 12 * progress;

        ctx.beginPath();
        ctx.strokeStyle = `rgba(0, 212, 255, ${0.05 + progress * 0.25})`;
        ctx.moveTo(0, y + wave);
        ctx.bezierCurveTo(
          width * 0.3, y - wave * 2,
          width * 0.7, y + wave * 2,
          width, y + wave
        );
        ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <div className="relative w-full h-full overflow-hidden bg-[#06090e]">
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
    </div>
  );
};
