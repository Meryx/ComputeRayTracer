import React, { useEffect } from "react";
import { Main } from "./main.js";

const App = () => {
  useEffect(() => {
    Main();
  }, []);
  return (
    <>
      <canvas id="canvas" width={700} height={700} />
    </>
  );
};

export default App;
