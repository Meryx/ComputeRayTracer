import React, { useEffect } from 'react';
import { Main } from './main.js';

const App = () => {
  useEffect(() => {
    Main();
  }, []);
  return (
    <>
      <canvas id="canvas" width={1920} height={1080} />
    </>
  );
};

export default App;
