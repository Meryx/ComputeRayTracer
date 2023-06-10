@group(0) @binding(0) var framebuffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage> primitives : array<Sphere>;
@group(0) @binding(2) var<storage, read_write> alt_color_buffer : array<vec3<f32>>;
@group(0) @binding(3) var<uniform> sample : u32;

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
  normal: vec3<f32>,
  t: f32,
  hit: bool,
  index: u32
};

var<private> light_source : vec3<f32> = vec3<f32>(0.0, 2.8, -0.2);
var<private> light_color : vec3<f32> = vec3<f32>(75.0, 75.0, 75.0);
var<private> light_center : vec3<f32> = vec3<f32>(0.0, 2.8, -0.2);
var<private> light_radius : f32 = 1.0;
var<private> albedo : array<vec3<f32>, 2>;

@compute 
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {

  albedo[0] = vec3<f32>(0.7, 0.7, 0.7);
  albedo[1] = vec3<f32>(0.7, 0.7, 0.0);

    let screen_size: vec2<u32> = textureDimensions(framebuffer);
    if(GlobalInvocationID.x >= screen_size.x || GlobalInvocationID.y >= screen_size.y) {
        return;
    }
    let screen_pos : vec2<i32> = vec2<i32>(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));

    seed = vec4<u32>(u32(screen_pos.x), u32(screen_pos.y * 100), sample, tea(u32(screen_pos.x), u32(screen_pos.y * 100)));
    
    var c : vec3<f32> = vec3<f32>(0);
    let ambient = vec3<f32>(0.08) * vec3<f32>(1.0);
    var shadow : bool = false;

    let lower_left_corner : vec3<f32> = vec3<f32>(-2.0, -1.0, 1.0);
    let horizontal : vec3<f32> = vec3<f32>(4.0, 0.0, 0.0);
    let vertical : vec3<f32> = vec3<f32>(0.0, 2.0, 0.0);
    let u : f32 = (f32(screen_pos.x) + 0.5) / f32(screen_size.x);
    let v : f32 = (f32(screen_size.y) - f32(screen_pos.y) + 0.5) / f32(screen_size.y);
    var ray : Ray;
    ray.origin = vec3<f32>(0.0, 0.0, 2.0);
    ray.direction = normalize(lower_left_corner + u * horizontal + v * vertical - ray.origin);
    var hit_record : HitRecord;
    hit_record.hit = false;
    hit_record.t = 100000.0;
    hit_record.index = 1u;
    light_source = sample_surface_circle(light_center, light_radius);
    for (var i = 0u; i < arrayLength(&primitives); i = i + 1u) {
      ray_sphere_intersection(primitives[i] , ray, &hit_record, i);
    }

    if(!hit_record.hit)
    {
      textureStore(framebuffer, screen_pos, vec4<f32>(c, 1.0));
      return;
    }
    for (var j = 0u; j < arrayLength(&primitives); j = j + 1u) {
      if(j == hit_record.index)
      {
        continue;
      }
      var shadow_hit_record : HitRecord;
      var shadow_ray : Ray;
      shadow_ray.origin = hit_record.p;
      shadow_ray.direction = normalize(light_source - hit_record.p);
      shadow_hit_record.hit = false;
      shadow_hit_record.t = 100000.0;
      ray_sphere_intersection(primitives[j] , shadow_ray, &shadow_hit_record, j);
      if(shadow_hit_record.hit)
      {
        c = c + vec3<f32>(0.08) * albedo[1];
        shadow = true;
        break;
      }
    }

    if(shadow)
    {
      let current_color = alt_color_buffer[u32(screen_pos.x) + u32(screen_pos.y) * screen_size.x];
      let combine = current_color + c;
      alt_color_buffer[u32(screen_pos.x) + u32(screen_pos.y) * screen_size.x] = combine;
      c = combine;
      c = c / f32(sample);
      c = c / (c + vec3<f32>(1.0));
      c = pow(c, vec3<f32>(1.0 / 2.2));
      textureStore(framebuffer, screen_pos, vec4<f32>(c, 1.0));
      return;
    }



    let normal : vec3<f32> = hit_record.normal;
    let light_dir = normalize(light_source - hit_record.p);
    let view_dir = normalize(ray.origin - hit_record.p);
    let h = normalize(light_dir + view_dir);
    let distance = length(light_source - hit_record.p);

    let exponent = 150.0f;
    let pref_direction = vec3<f32>(0.0, -1.0, 0.0);
    let attuenation = pow((max(dot(normalize(-pref_direction), light_dir), 0.0)), exponent) / (distance * distance);
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

    let F0 = mix(vec3<f32>(0.04), albedo[hit_record.index], metallic);
    let F = F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - dot(h, view_dir), 0.0, 1.0), 5.0);

    let numerator = NDF * G * F;
    let denominator = 4.0 * max(ndotv, 0.0) * max(ndotl, 0.0) + 0.0001;

    let specular = numerator / denominator;
    let kS = F;
    var kD = vec3<f32>(1.0) - kS;
    kD = kD * (1.0 - metallic);

    let Lo = ((kD * albedo[hit_record.index] / PI + specular) * radiance * ndotl);
    
    c = c + Lo + ambient;

    c = c * albedo[hit_record.index];

    let current_color = alt_color_buffer[u32(screen_pos.x) + u32(screen_pos.y) * screen_size.x];
    let combine = current_color + c;
    alt_color_buffer[u32(screen_pos.x) + u32(screen_pos.y) * screen_size.x] = combine;

    c = combine;
    c = c / f32(sample);
    
    c = c / (c + vec3<f32>(1.0));
    c = pow(c, vec3<f32>(1.0 / 2.2));
    textureStore(framebuffer, screen_pos, vec4<f32>(c, 1.0));
    
}

fn ray_at(ray : Ray, t : f32) -> vec3<f32>
{
  return ray.origin + t * ray.direction;
}

/* Primitive intersection */
fn ray_sphere_intersection(sphere : Sphere, ray : Ray, hit_record : ptr<function, HitRecord>, index : u32)
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
    return;
  }

  var t : f32 = (-b - sqrt(discriminant)) / (2.0 * a);
  if (t < 0.0 || t > (*hit_record).t) {
    t = (-b + sqrt(discriminant)) / (2.0 * a);
    if(t < 0.0 || t > (*hit_record).t)
    {
      return;
    }
  }

  (*hit_record).p = ray_at(ray, t);
  (*hit_record).normal = normalize((*hit_record).p - center);
  (*hit_record).t = t;
  (*hit_record).hit = true;
  (*hit_record).index = index;
}


/* Sampling */
fn sample_circle(radius : f32) -> vec2<f32>
{
    let r = sqrt(rand()) * radius;
    let theta = rand() * 2.0 * PI;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

fn sample_surface_circle(center : vec3<f32>, radius : f32) -> vec3<f32>
{
    let offset = sample_circle(radius);
    return vec3<f32>(center.x + offset.x, center.y , center.z + offset.y);
}


fn sample_circle_cosine_weighted(radius : f32) -> vec2<f32>
{
  let r = sqrt(rand()) * radius;
  let theta = rand() * 2.0 * PI;

  let u = rand();
  let v = rand();
  let phi = 2.0 * PI * u;
  let cosTheta = sqrt(1.0 - v);
  let sinTheta = sqrt(v);

  let x = cos(phi) * sinTheta;
  let y = sin(phi) * sinTheta;
  return vec2<f32>(r * x, r * y);
}


/* Utilties to generate random numbers */

fn tea(val0:u32, val1:u32)->u32{
// "GPU Random Numbers via the Tiny Encryption Algorithm"
  var v0 = val0;
  var v1 = val1;
  var s0 = u32(0);
  for (var n: i32 = 0; n < 16; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

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