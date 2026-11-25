import React from 'react';

interface FlashTransitionProps {
  triggerKey: number;
}

export const FlashTransition: React.FC<FlashTransitionProps> = ({ triggerKey }) => {
  if (triggerKey === 0) return null;

  return (
    <div
      key={triggerKey}
      className="flash-white-overlay"
      aria-hidden="true"
    />
  );
};
