#[macro_use]
extern crate glium;
extern crate image;
extern crate rand;
extern crate glium_text_rusttype as text;
extern crate csv;

use glium::{Surface, glutin};
use glium::index::PrimitiveType;
use std::env;
use std::thread;
use std::fs::File;
use std::path::Path;
use rand::{Rng, thread_rng};
use std::sync::{Arc};
use std::sync::atomic::{AtomicBool, Ordering};


fn load_csv() -> Vec<csv::StringRecord>{
    let f = File::open(&Path::new("/Users/goncalopalaio/Dropbox/2017/kaggle-facial-keypoints-detection/dataset/training.csv")).expect("Error reading csv");
    let mut reader = csv::Reader::from_reader(f);
    let mut vec: Vec<csv::StringRecord> = Vec::new();

    for r in reader.records() {
        //println!("{:?}", r);

        match r {
            Ok(record) => vec.push(record),
            _ => (),
        }
    }

    vec
}
fn create_texture(display: &glium::Display, image_path: &str) -> glium::Texture2d {
    let image = image::open(image_path).unwrap().to_rgba();
    let image_dims = image.dimensions();
    let image = glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dims);
    glium::texture::Texture2d::new(display, image).unwrap()
}

/// Builds a 2x2 unicolor texture.
fn build_unicolor_texture2d(display: &glium::Display, red: f32, green: f32, blue: f32) -> glium::Texture2d {
    let color = ((red * 255.0) as u8, (green * 255.0) as u8, (blue * 255.0) as u8);

    glium::texture::Texture2d::new(display, vec![
        vec![color, color],
        vec![color, color],
    ]).unwrap()
}

fn blit_record(/*record : &csv::StringRecord, */marker_texture: &glium::Texture2d,  other_texture: &glium::Texture2d) {
    /*let left_eye_center_x = match record[0].parse::<f32>() {
        Ok(value) => value,
        _ => 0.0
    } as u32;
    let left_eye_center_y = match record[1].parse::<f32>() {
        Ok(value) => value,
        _ => 0.0
    } as u32;
*/

    let mut random_gen = thread_rng();

    let left_eye_center_x = random_gen.gen_range(0, 90);
    let left_eye_center_y = random_gen.gen_range(0, 90);

//    println!("left_eye_center_x {:?} left_eye_center_y {:?}", left_eye_center_x, left_eye_center_y);

    marker_texture.as_surface().blit_whole_color_to(&other_texture.as_surface(), &glium::BlitTarget {
                    left: left_eye_center_x,
                    bottom: left_eye_center_y,
                    width: 6,
                    height: 6,
                }, glium::uniforms::MagnifySamplerFilter::Linear);

}

fn main() {
    println!("Hello, world!");

    println!("Starting data loader thread");

    let exiting_program = Arc::new(AtomicBool::new(false));
    let screen_is_dirty = Arc::new(AtomicBool::new(false));

    let exiting_program_clone = exiting_program.clone();
    let screen_is_dirty_clone = screen_is_dirty.clone();
    let data_thread = thread::spawn(move || {

        loop {
                let point_records = load_csv();        
                println!("Finished loading {:?}", point_records.len());
                screen_is_dirty_clone.store(true, Ordering::SeqCst);
                thread::park();

                if exiting_program_clone.load(Ordering::SeqCst) {
                    break;
                }
        }
        
    });
    

    let w = 1000;
    let h = 1000;

 	let version = parse_version();
 	println!("{:?}", version);
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new().with_dimensions(w, h);
    let context = glutin::ContextBuilder::new()
        .with_gl_debug_flag(true)
        .with_gl(version);
    let display = glium::Display::new(window, context, &events_loop).unwrap();
    let (vb, ib) = build_rectangle_vb_ib(&display);

    let program = glium::Program::from_source(&display,
        "

            #version 140

            uniform mat4 matrix;

            in vec2 position;
            in vec2 tex_coords;

            out vec2 v_tex_coords;

            void main() {
                gl_Position = matrix * vec4(position, 0.0, 1.0);
                v_tex_coords = tex_coords;
            }
        ",
        "
            #version 140

            uniform sampler2D tex;
            in vec2 v_tex_coords;
            out vec4 color;

            void main() {
                color = texture(tex, v_tex_coords);
            }
        ",
        None).expect("Error building shader program");
    
        //image::JPEG).unwrap().to_rgba();

    let main_texture = glium::Texture2d::empty(&display, 1024, 1024).unwrap();

    main_texture.as_surface().clear_color(1.0, 0.0, 1.0, 1.0);

    let marker_texture = build_unicolor_texture2d(&display, 1.0, 0.0, 0.0);

    
    let text_system = text::TextSystem::new(&display);
    let font_file = File::open(&Path::new(&"font.ttf")).expect(".ttf file not found");
    let font = text::FontTexture::new(&display,
                                      font_file,
                                      30,
                                      text::FontTexture::ascii_character_list()).unwrap();
    let text_matrix: [[f32; 4]; 4] = [
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0,],
                [0.0, 0.0, 0.1, 0.0],
                [-0.1, 0.0, 0.0, 1.0f32]];

    let mut text = text::TextDisplay::new(&text_system, &font, &"Hello");

    let mut frame_count = 0;
    let mut stop_drawing = false;
    while !stop_drawing {
        frame_count += 1;


        if screen_is_dirty.load(Ordering::SeqCst) {
            println!("Reloading");
            // Load all textures
            let mut img_idx = 0;
            for i in 0..10 {
                for j in 0..10 {
                    let other_texture = create_texture(&display,&format!("../../../images/train/{}_img.png", img_idx));    
                    let dest_rect = glium::BlitTarget {
                            left: 0 + i * 96,
                            bottom: 900 - j * 96,
                            width: 96,
                            height: 96,
                        };

                    //let curr_record = &point_records[img_idx];
                    //println!("curr_record {} {:?}", img_idx, curr_record);
                    /*left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,left_eye_inner_corner_x,left_eye_inner_corner_y,left_eye_outer_corner_x,left_eye_outer_corner_y,right_eye_inner_corner_x,right_eye_inner_corner_y,right_eye_outer_corner_x,right_eye_outer_corner_y,left_eyebrow_inner_end_x,left_eyebrow_inner_end_y,left_eyebrow_outer_end_x,left_eyebrow_outer_end_y,right_eyebrow_inner_end_x,right_eyebrow_inner_end_y,right_eyebrow_outer_end_x,right_eyebrow_outer_end_y,nose_tip_x,nose_tip_y,mouth_left_corner_x,mouth_left_corner_y,mouth_right_corner_x,mouth_right_corner_y,mouth_center_top_lip_x,mouth_center_top_lip_y,mouth_center_bottom_lip_x,mouth_center_bottom_lip_y,Image*/

                    blit_record(&marker_texture, &other_texture);
                    
                    other_texture.as_surface().blit_whole_color_to(&main_texture.as_surface(), &dest_rect,
                                                                        glium::uniforms::MagnifySamplerFilter::Linear);

                    img_idx += 1;
                }
            }
            screen_is_dirty.store(false, Ordering::SeqCst);
        }
        


    
    let uniforms = uniform! {
        matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0f32]],
        tex: &main_texture,

    };
        //main_texture.as_surface().draw(&vb, &ib, &program, &uniforms, &Default::default()).unwrap();    
        let mut target = display.draw();

            target.clear_color(0.0, 1.0, 0.1, 1.0);
            
            target.draw(&vb, &ib, &program, &uniforms, &Default::default()).unwrap();

            text.set_text(&format!("Hello |{}", frame_count));
            text::draw(&text, &text_system, &mut target, text_matrix, (1.0,0.0,0.0,1.0)).unwrap();


        target.finish().unwrap();

         // drawing a frame
        //let target = display.draw();
        //dest_texture.as_surface().fill(&target, glium::uniforms::MagnifySamplerFilter::Linear);
        //target.finish().unwrap();



        events_loop.poll_events(|ev| {
            match ev {
                glutin::Event::WindowEvent {event, ..} => match event {
                    glutin::WindowEvent::Closed => stop_drawing = true,
                    glutin::WindowEvent::KeyboardInput {input, ..} => {
                    
                        match input.virtual_keycode {
                            Some (keycode) => {
                                match keycode {
                                    glutin::VirtualKeyCode::Escape => {
                                        println!("Escape!");
                                        stop_drawing = true;
                                        exiting_program.store(true, Ordering::SeqCst);
                                        data_thread.thread().unpark();
                                    },
                                    glutin::VirtualKeyCode::Space if input.state == glutin::ElementState::Released => {
                                        println!("Space pressed");
                                        data_thread.thread().unpark();
                                    },
                                    _ => (),
                                }
                            }
                            _ => (),
                        }
                    }
                    _ => (),
                },
                _ => (),

            }
        });
        
    };
    
    let _ = data_thread.join();

    println!("Exiting");
}

fn build_rectangle_vb_ib(display: &glium::Display)
    -> (glium::vertex::VertexBufferAny, glium::index::IndexBufferAny)
{
    #[derive(Copy, Clone)]
    struct Vertex {
        position: [f32; 2],
        tex_coords: [f32; 2],

    }

    implement_vertex!(Vertex, position, tex_coords);

    (
        glium::VertexBuffer::new(display, &[
        Vertex { position: [-1.0, -1.0], tex_coords: [0.0,  0.0] },
        Vertex { position: [ -1.0, 1.0], tex_coords: [0.0,  1.0] },
        Vertex { position: [ 1.0,  -1.0], tex_coords: [1.0,  0.0] },
        Vertex { position: [1.0,  1.0], tex_coords: [1.0,  1.0] },
        ]).unwrap().into(),

        glium::IndexBuffer::new(display, PrimitiveType::TrianglesList, &[0u8, 1, 2, 2, 1, 3]).unwrap().into()

    )
}

fn parse_version() -> glutin::GlRequest {
    match env::var("GLIUM_GL_VERSION") {
        Ok(version) => {
            // expects "OpenGL 3.3" for example

            let mut iter = version.rsplitn(2, ' ');

            let version = iter.next().unwrap();
            let ty = iter.next().unwrap();

            let mut iter = version.split('.');
            let major = iter.next().unwrap().parse().unwrap();
            let minor = iter.next().unwrap().parse().unwrap();

            let ty = if ty == "OpenGL" {
                glutin::Api::OpenGl
            } else if ty == "OpenGL ES" {
                glutin::Api::OpenGlEs
            } else if ty == "WebGL" {
                glutin::Api::WebGl
            } else {
                panic!();
            };

            glutin::GlRequest::Specific(ty, (major, minor))
        },
        Err(_) => glutin::GlRequest::Latest,
    }
}

