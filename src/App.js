import React, { useEffect } from 'react';
import { Main } from './main.js';

const App = () => {
  useEffect(() => {
    Main();
  }, []);
  return (
    <>
      <canvas id="canvas" width={1366} height={768} />
    </>
  );
};

export default App;
