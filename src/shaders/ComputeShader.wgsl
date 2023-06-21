@group(0) @binding(0) var framebuffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage> primitives : array<Sphere>;
@group(0) @binding(2) var<storage, read_write> alt_color_buffer : array<vec3<f32>>;
@group(0) @binding(3) var<uniform> sample : u32;
@group(0) @binding(4) var<storage> planar_patches : array<PlanarPatch>;
@group(0) @binding(5) var<storage> camera : Camera;


const PI: f32 = 3.14159265359;
const roughness: f32 = 0.4;
const metallic: f32 = 0.1;
const MAXDEPTH : u32 = 100;
const INFINITY : f32 = 3.40282346638528859812e+38f;
const EMISSION : vec3<f32> = vec3<f32>(2, 2, 2);
const area : f32 = 100 * 100;
const grid : u32 = 1;

struct Sphere {
    geometry: vec4<f32>,
    albedo: vec3<f32>,
    index: u32
};

struct Camera {
  origin: vec3<f32>,
  direction: vec3<f32>,
  focal_distance_width_height: vec3<f32>
}

struct PlanarPatch {
  origin: vec3<f32>,
  edge1: vec3<f32>,
  edge2: vec3<f32>,
  albedo: vec3<f32>,
  emission: vec3<f32>,
  index: u32
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
  index: u32,
  albedo: vec3<f32>,
  emission: vec3<f32>,
  last_index: u32,
};

struct ShapeIntersection {
  position: vec3<f32>,
  normal: vec3<f32>,
  t: f32,
  hit: bool,
  index: u32,
  albedo: vec3<f32>,
  emission: vec3<f32>,
  last_index: u32,
};

@compute 
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {

    let screen_size: vec2<u32> = textureDimensions(framebuffer);
    if(GlobalInvocationID.x >= screen_size.x || GlobalInvocationID.y >= screen_size.y) {
        return;
    }
    let screen_pos : vec2<i32> = vec2<i32>(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));
    seed = vec4<u32>(screen_size.y - u32(screen_pos.y), screen_size.x - u32(screen_pos.x), sample, tea(screen_size.x - u32(screen_pos.x), screen_size.y - u32(screen_pos.y)));
    let brdf = 1 / PI;

    let lower_left_corner : vec3<f32> = camera.origin + camera.focal_distance_width_height;
    let horizontal : vec3<f32> = -vec3<f32>(camera.focal_distance_width_height.x * 2, 0, 0);
    let vertical : vec3<f32> = -vec3<f32>(0, camera.focal_distance_width_height.y * 2, 0);
    var color : vec3<f32> = vec3<f32>(0.0);

      let u : f32 = (f32(screen_pos.x) + (f32(sample % grid) + rand()) / f32(grid)) / f32(screen_size.x);
      let v : f32 = (f32(screen_size.y) - f32(screen_pos.y) + (f32(sample % grid) + rand()) / f32(grid)) / f32(screen_size.y);

      var ray : Ray;
      ray.origin = camera.origin;
      ray.direction = normalize(lower_left_corner + u * horizontal + v * vertical - ray.origin);

      var depth : u32 = 0;
      var radiance : vec3<f32> = vec3<f32>(0.0);
      var beta : vec3<f32> = vec3<f32>(1.0);
      var pdf_b : f32 = 1.0;
      var prev_intersection : ShapeIntersection;
      while(true)
      {

        let shape_intersection : ShapeIntersection = intersect(ray);
        if(!shape_intersection.hit)
        {
          break;
        }


        let radiance_emitted = shape_intersection.emission;
        if(length(radiance_emitted) > 0)
        {
          if(depth == 0)
          {
            radiance += beta * EMISSION;
          } else {
            let pdf_l = (1.0 / area) / (abs(dot(shape_intersection.normal, -ray.direction)) / pow(length(prev_intersection.position - shape_intersection.position), 2));
            let weight_b = power_heuristic(1, pdf_b, 1, pdf_l);
            var to_add = beta * weight_b * EMISSION;
            if(!(length(to_add) == length(to_add)))
            {
              to_add = vec3<f32>(0,0,0);
            }
            radiance += to_add;
          }
          break;
        }

        if(depth >= MAXDEPTH)
        {
          break;
        }

        let light_normal = vec3<f32>(0,-1,0);
        beta *= shape_intersection.albedo;


        let point_on_light = sample_light();
        let is_visible = is_visible(shape_intersection.position, point_on_light);
        let light_dir = normalize(point_on_light - shape_intersection.position);
        let radiance_light = is_visible * EMISSION * max(0,(dot(shape_intersection.normal, light_dir)));
        var pdf_l = (1.0 / area) / (abs(dot(light_normal, -light_dir)) / (pow(length(shape_intersection.position - point_on_light), 2)));



        var theta = abs(dot(shape_intersection.normal, light_dir));
        if(theta <= 0.0001)
        {
          theta = 0;
        }
        pdf_b = cos(theta) / PI;
        let weight_l = power_heuristic(1, pdf_l, 1, pdf_b);
        var to_add = brdf * beta * radiance_light * (weight_l / pdf_l);

        if(!(length(to_add) == length(to_add)))
        {
          to_add = vec3<f32>(0);
        }


        radiance += to_add;

        let new_direction_with_pdf = cosine_weighted_sample_hemisphere(shape_intersection.normal);
        let new_direction = new_direction_with_pdf.xyz;
        var pdf = new_direction_with_pdf.w;
        beta *= brdf * abs(dot(shape_intersection.normal, normalize(new_direction))) / pdf;
        pdf_b = pdf;
        prev_intersection = shape_intersection;

        ray.origin = shape_intersection.position;
        ray.direction = normalize(new_direction);
        depth++;
      }
      if(!(length(radiance) == length(radiance)))
      {
        radiance = vec3<f32>(0);
      }

      color = radiance;



    if(!(length(color) == length(color)))
    {
      color = vec3<f32>(1);
    }




    var c = color;
    var current_color = alt_color_buffer[u32(screen_pos.x) + u32(screen_pos.y) * screen_size.x];
    let combine = current_color + c;
    alt_color_buffer[u32(screen_pos.x) + u32(screen_pos.y) * screen_size.x] = combine;
    c = combine;
    c = c / f32(sample);
    c = c / (c + vec3<f32>(1.0));
    c = pow(c, vec3<f32>(1.0 / 2.2));
    if(!(length(c) == length(c)))
    {
      return;
    }
    textureStore(framebuffer, screen_pos, vec4<f32>(c, 1.0));
    
}

fn is_visible(p0 : vec3<f32>, p1 : vec3<f32>) -> f32
{
  var ray : Ray;
  ray.origin = p0;
  ray.direction = normalize(p1 - p0);
  let shape_intersection : ShapeIntersection = intersect(ray);
  if(shape_intersection.hit)
  {
    if(length(shape_intersection.position - p1) < 0.001 && shape_intersection.index == 2)
    {
      return 1.0;
    }
  }
  return 0.0;
}

fn sample_light() -> vec3<f32>
{
  let light = planar_patches[2];
  let u = rand();
  let v = rand();
  let p = light.origin + u * light.edge1 + v * light.edge2;
  return p;
}

fn power_heuristic(nf : f32, f_pdf : f32, ng : f32, g_pdf : f32) -> f32
{
  let f = nf * f_pdf;
  let g = ng * g_pdf;
  return (f * f) / (f * f + g * g);
}

fn intersect(ray : Ray) -> ShapeIntersection
{
  var shape_intersection : ShapeIntersection;
  shape_intersection.hit = false;
  var t_max : f32 = INFINITY;

  for(var i : u32; i < arrayLength(&planar_patches); i++)
  {
    let temp_si = ray_patch_intersection_test(planar_patches[i], ray, 0.001, t_max);
    if(temp_si.hit)
    {
      shape_intersection = temp_si;
      t_max = temp_si.t;
    }
    
  }
  return shape_intersection;
}

fn ray_at(ray : Ray, t : f32) -> vec3<f32>
{
  return ray.origin + t * ray.direction;
}

/* Primitive intersection */
fn ray_sphere_intersection(sphere : Sphere, ray : Ray, hit_record : ptr<function, HitRecord>)
{
  if((*hit_record).last_index == sphere.index)
  {
    return;
  }

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
  (*hit_record).index = sphere.index;
  (*hit_record).albedo = sphere.albedo;
}

fn ray_patch_intersection_test(planar_patch : PlanarPatch, ray : Ray, t_min : f32, t_max : f32) -> ShapeIntersection
{

  var shape_intersection : ShapeIntersection;
  shape_intersection.hit = false;


  let origin = ray.origin;
  let direction = ray.direction;
  let edge1 = planar_patch.edge1;
  let edge2 = planar_patch.edge2;
  var normal = normalize(cross((edge1), (edge2)));
  let origin_to_origin = planar_patch.origin - origin;
  var ndotd = dot(normal, direction);
  if(ndotd > 0)
  {
    normal = -normal;
  }
  ndotd = dot(normal, direction);
  if(abs(ndotd)  < 0.0001)
  {
    return shape_intersection;
  }

  let t = dot(normal, origin_to_origin) / (ndotd);
  if(t < t_min || t > t_max)
  {
    return shape_intersection;
  }
  let intersection = ray_at(ray, t);
  let m = intersection - planar_patch.origin;
  let u = dot(m, edge1) / dot(edge1, edge1);
  let v = dot(m, edge2) / dot(edge2, edge2);

  if(u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
  {
    return shape_intersection;
  }

  shape_intersection.hit = true;
  shape_intersection.position = intersection;
  shape_intersection.normal = normalize(normal);
  shape_intersection.t = t;
  shape_intersection.index = planar_patch.index;
  shape_intersection.albedo = planar_patch.albedo;
  shape_intersection.emission = planar_patch.emission;
  return shape_intersection;
}


fn ray_patch_intersection(planar_patch : PlanarPatch, ray : Ray, hit_record : ptr<function, HitRecord>)
{


  let origin = ray.origin;
  let direction = ray.direction;
  let edge1 = planar_patch.edge1;
  let edge2 = planar_patch.edge2;
  var normal = normalize(cross(normalize(edge1), normalize(edge2)));
  let origin_to_origin = planar_patch.origin - origin;
  var ndotd = dot(normal, direction);
  if(ndotd > 0)
  {
    normal = -normal;
  }
  ndotd = dot(normal, direction);
  if(abs(ndotd)  < 0.0001)
  {
    return;
  }

  let t = dot(normal, origin_to_origin) / (ndotd);
  if(t < 0.0 || t > (*hit_record).t)
  {
    return;
  }
  let intersection = ray_at(ray, t);
  let m = intersection - planar_patch.origin;
  let u = dot(m, edge1) / dot(edge1, edge1);
  let v = dot(m, edge2) / dot(edge2, edge2);

  if(u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
  {
    return;
  }

  (*hit_record).hit = true;
  (*hit_record).p = intersection;
  (*hit_record).normal = normalize(normal);
  (*hit_record).t = t;
  (*hit_record).index = planar_patch.index;
  (*hit_record).albedo = planar_patch.albedo;
  (*hit_record).emission = planar_patch.emission;
}


/* Sampling */

fn uniformally_sample_hemisphere(normal: vec3<f32>) -> vec3<f32> {
  let u = rand();
  let v = rand();
  let theta = 2.0 * PI * u;
  let phi = acos(2.0 * v - 1.0);
  let x = cos(theta) * sin(phi);
  let y = sin(theta) * sin(phi);
  let z = cos(phi);
  var sample = vec3<f32>(x, y, z);
  if(dot(sample, normal) < 0.0)
  {
    sample = -sample;
  }
  return sample;
}

fn cosine_weighted_sample_hemisphere(normal : vec3<f32>) -> vec4<f32> {
  let u = rand();
  let v = rand();
  let r = sqrt(u);
  let theta = 2.0 * PI * v;
  let x = r * cos(theta);
  let y = r * sin(theta);
  let z = sqrt(max(0.0, 1.0 - u));

  var up : vec3<f32>;
  if(abs(normal.z) < 0.999)
  {
    up = vec3<f32>(0.0, 0.0, 1.0);
  }
  else
  {
    up = vec3<f32>(1.0, 0.0, 0.0);
  }
  let tangent = normalize(cross(up, normal));
  let bitangent = cross(normal, tangent);
  let direction = tangent * x + bitangent * y + normal * z;
  return vec4<f32>(direction, z/PI);
}


fn uniformally_sample_rectangle(rect : PlanarPatch) -> vec3<f32>
{
  let u = rand();
  let v = rand();
  let point = rect.origin + u * rect.edge1 + v * rect.edge2;
  return point;
}

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
    return f32(seed.x & 0x00ffffffu) / f32(0x00ffffffu);
}

var<private> seed : vec4<u32> = vec4<u32>(0);