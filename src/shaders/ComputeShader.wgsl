@group(0) @binding(0) var framebuffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage> primitives : array<Sphere>;

const PI: f32 = 3.14159265359;
const roughness: f32 = 0.4;
const metallic: f32 = 0.1;

struct Sphere {
    geometry: vec4<f32>
};

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
};

struct HitRecord {
  p: vec3<f32>,
  normal: vec3<f32>
};

var<private> light_source : vec3<f32> = vec3<f32>(0.0, 2.8, -1.6);
var<private> light_color : vec3<f32> = vec3<f32>(5.0, 5.0, 5.0);

@compute 
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {

    let screen_size: vec2<u32> = textureDimensions(framebuffer);
    if(GlobalInvocationID.x >= screen_size.x || GlobalInvocationID.y >= screen_size.y) {
        return;
    }
    let screen_pos : vec2<i32> = vec2<i32>(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));

    seed = vec4<u32>(u32(screen_pos.x), u32(screen_pos.y), 2u, u32(screen_pos.x) + u32(screen_pos.y));
    
    var c : vec3<f32> = vec3<f32>(0);

    let lower_left_corner : vec3<f32> = vec3<f32>(-2.0, -1.0, 1.0);
    let horizontal : vec3<f32> = vec3<f32>(4.0, 0.0, 0.0);
    let vertical : vec3<f32> = vec3<f32>(0.0, 2.0, 0.0);
    let u : f32 = (f32(screen_pos.x) + 0.5) / f32(screen_size.x);
    let v : f32 = (f32(screen_size.y) - f32(screen_pos.y) + 0.5) / f32(screen_size.y);
    var ray : Ray;
    ray.origin = vec3<f32>(0.0, 0.0, 2.0);
    ray.direction = normalize(lower_left_corner + u * horizontal + v * vertical - ray.origin);
    for (var i = 0u; i < arrayLength(&primitives); i = i + 1u) {
        var hit_record : HitRecord;
        if (ray_sphere_intersection(primitives[i] , ray, &hit_record)) {
            let normal : vec3<f32> = hit_record.normal;

            let light_dir = normalize(light_source - hit_record.p);
            let view_dir = normalize(ray.origin - hit_record.p);
            let h = normalize(light_dir + view_dir);
            let distance = length(light_source - hit_record.p);
            let attuenation = 1.0 / (distance * distance);
            let radiance = light_color * attuenation;

            //Cook-Torrance BRDF

            //First distrubtionGGX
            let a = roughness * roughness;
            let a2 = a * a;
            let ndoth = max(dot(normal, h), 0.0);
            let ndoth2 = ndoth * ndoth;
            let nom = a2;
            let denom = (ndoth2 * (a2 - 1.0) + 1.0);
            let d = nom / (PI * denom * denom);
            let NDF = d;


            //Next Geometry Smith
            let ndotv = max(dot(normal, view_dir), 0.0);
            let ndotl = max(dot(normal, light_dir), 0.0);
            
            let r = roughness + 1.0;
            let k = (r * r) / 8.0;
            let ggx2 = ndotv / (ndotv * (1.0 - k) + k);
            let ggx1 = ndotl / (ndotl * (1.0 - k) + k);
            let G = ggx1 * ggx2;

            let F0 = mix(vec3<f32>(0.04), vec3<f32>(0.7), metallic);
            let F = F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - dot(h, view_dir), 0.0, 1.0), 5.0);

            let numerator = NDF * G * F;
            let denominator = 4.0 * max(ndotv, 0.0) * max(ndotl, 0.0) + 0.0001;

            let specular = numerator / denominator;
            let kS = F;
            var kD = vec3<f32>(1.0) - kS;
            kD = kD * (1.0 - metallic);

            let Lo = (kD * vec3<f32>(0.7) / PI + specular) * radiance * ndotl;

            let ambient = vec3<f32>(0.02) * vec3<f32>(0.7);
            
            c = c + Lo + ambient;
            c = c / (c + vec3<f32>(1.0));
            c = pow(c, vec3<f32>(1.0 / 2.2));
        }
    }
    textureStore(framebuffer, screen_pos, vec4<f32>(c, 1.0));
    
}

fn ray_at(ray : Ray, t : f32) -> vec3<f32>
{
  return ray.origin + t * ray.direction;
}

/* Primitive intersection */
fn ray_sphere_intersection(sphere : Sphere, ray : Ray, hit_record : ptr<function, HitRecord>) -> bool
{
  let origin = ray.origin;
  let direction = ray.direction;
  let radius = sphere.geometry.w;
  let center = vec3<f32>(sphere.geometry.xyz);
  let oc = origin - center;
  let a = dot(direction, direction);
  let b = 2.0 * dot(oc, direction);
  let c = dot(oc, oc) - radius * radius;
  let discriminant = b * b - 4.0 * a * c;

  if (discriminant < 0.0) {
    return false;
  }

  var t : f32 = (-b - sqrt(discriminant)) / (2.0 * a);
  if (t < 0.0) {
    t = (-b + sqrt(discriminant)) / (2.0 * a);
    if(t < 0.0)
    {
      return false;
    }
  }

  (*hit_record).p = ray_at(ray, t);
  (*hit_record).normal = normalize((*hit_record).p - center);
  return true;
}


/* Utilties to generate random numbers */
fn pcg4d()
{
    seed = seed * 1664525u + 1013904223u;
    seed.x += seed.y * seed.w; 
    seed.y += seed.z * seed.x; 
    seed.z += seed.x * seed.y; 
    seed.w += seed.y * seed.z;
    seed = seed ^ (seed >> vec4<u32>(16u, 16u, 16u, 16u));
    seed.x += seed.y * seed.w; 
    seed.y += seed.z * seed.x; 
    seed.z += seed.x * seed.y; 
    seed.w += seed.y * seed.z;
}

fn rand() -> f32
{
    pcg4d(); 
    return f32(seed.x) / f32(0xffffffffu);
}

var<private> seed : vec4<u32> = vec4<u32>(0);