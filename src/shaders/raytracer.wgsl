@group(0) @binding(0) var colorBuffer: texture_storage_2d<rgba8unorm, write>;

/* Numerical constants */
const PI: f32 = 3.14159265359;
const INFINITY : f32 = 3.40282346638528859812e+3;

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

fn lengthSquared(v : vec3<f32>) -> f32 {
    return dot(v, v);
}

fn degreesToRadians(angle : f32) -> f32
{
    return angle * PI / 180;
}

fn isEqual(v1 : vec3<f32>, v2 : vec3<f32>) -> bool
{
    return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
}

@compute @workgroup_size(1,1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let screenSize: vec2<u32> = textureDimensions(colorBuffer);
    let screenPos : vec2<i32> = vec2<i32>(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));

    var ray : Ray;
    var pixelColor : vec3<f32> = vec3<f32>(0, 0, 0);

    let u : f32 = (f32(screenPos.x) + 0.5) / f32(screenSize.x - 1);
    let v : f32 = (f32(screenSize.y) - f32(screenPos.y) + 0.5) / f32(screenSize.y - 1);

    let lowerLeftCorner = vec3<f32>(-1.5, -1.0, 0.0);
    let horizontal = vec3<f32>(3.0, 0.0, 0.0);
    let vertical = vec3<f32>(2.0, 0.0, 0.0);

    ray.origin = vec3<f32>(0.0, 0.0, 1.0);
    ray.direction = lowerLeftCorner + u * horizontal + v * vertical - ray.origin;

    let grad = normalize(ray.direction).y;
    let t2 = 0.5 * (grad + 1.0);
    pixelColor = (1.0 - t2) * vec3<f32>(1.0, 1.0, 1.0) + t2 * vec3<f32>(0.5, 0.7, 1.0);
    textureStore(colorBuffer, screenPos, vec4<f32>(pixelColor, 1.0));
}