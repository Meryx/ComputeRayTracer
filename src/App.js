import React, { useEffect } from "react";
import { Main } from "./main.js";
import cornell from "./scenes/cornell.json";

const App = () => {
  useEffect(() => {
    Main();
  }, []);
  return (
    <>
      <canvas
        id="canvas"
        width={cornell.camera.width}
        height={cornell.camera.height}
      />
    </>
  );
};

export default App;
