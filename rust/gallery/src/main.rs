extern crate image;
extern crate rand;
extern crate csv;
extern crate minifb;

use std::fs::File;
use std::path::Path;
use minifb::{MouseButton, MouseMode, Window, Key, Scale, WindowOptions};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;



fn load_csv() -> Vec<csv::StringRecord>{
    let f = File::open(&Path::new("/Users/goncalopalaio/Dropbox/2017/kaggle-facial-keypoints-detection/python/test_results.csv")).expect("Error reading csv");
    let mut reader = csv::Reader::from_reader(f);
    let mut vec: Vec<csv::StringRecord> = Vec::new();

    for r in reader.records() {
        println!("{:?}", r);

        match r {
            Ok(record) => vec.push(record),
            _ => (),
        }
    }

    vec
}

fn paste_image(buffer: &mut Vec<u32>, image_index: u32, point_list: &Vec<csv::StringRecord>) {
    let image = image::open(format!("/Users/goncalopalaio/Dropbox/2017/kaggle-facial-keypoints-detection/images/test/{}_img.png", image_index)).unwrap().to_rgba();
    let points = point_list.get(image_index as usize).unwrap();
    //println!("current points: {:?}", points);
    let offset_x = image_index * 96;
    let offset_y = image_index * 96;
    
    for i in 0..96 {
        for j in 0..96 {            
            //buffer[i + j * WIDTH] = 0x00ffffff
            let index = ((i + offset_x) + (j + offset_y) * WIDTH as u32) as usize;
            buffer[index] = image[(i as u32,j as u32)][0] as u32;
        }
    }

    let mut r = 0;
    while r < 30 - 1 {
        
        let x = points.get(r).unwrap().parse::<usize>().unwrap() + (offset_x as usize);
        let y = points.get(r + 1).unwrap().parse::<usize>().unwrap() + (offset_y as usize);

        buffer[x + y * WIDTH] = 0x00ffffff;
        r += 2;
    }

}

fn main() {
    println!("Hello, world!");

    let points = load_csv();
    //let image = image::open(image_path).unwrap().to_rgba();
    //let image_dims = image.dimensions();
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];

    let mut window = match Window::new("Gallery - Press ESC to exit", WIDTH, HEIGHT,
                                       WindowOptions {
                                           scale: Scale::X2,
                                           ..WindowOptions::default()
                                       }) {
        Ok(win) => win,
        Err(err) => {
            println!("Unable to create window {}", err);
            return;
        }
    };

    while window.is_open() && !window.is_key_down(Key::Escape) {
        paste_image(&mut buffer, 0, &points);
        paste_image(&mut buffer, 1, &points);
        paste_image(&mut buffer, 2, &points);

        window.get_keys().map(|keys| {
            for t in keys {
                match t {
                    Key::W => println!("holding w!"),
                    Key::T => println!("holding t!"),
                    _ => (),
                }
            }
        });

        window.get_mouse_pos(MouseMode::Discard).map(|mouse| {
            let screen_pos = ((mouse.1 as usize) * WIDTH) + mouse.0 as usize;
            
            if window.get_mouse_down(MouseButton::Left) {
                println!("{:?}", window.get_unscaled_mouse_pos(MouseMode::Discard).unwrap());
                buffer[screen_pos] = 0x00ffffff;
            }

            if window.get_mouse_down(MouseButton::Right) {
                buffer[screen_pos] = 0;
            }
        });

        window.get_scroll_wheel().map(|scroll| {
            println!("Scrolling {} - {}", scroll.0, scroll.1);
        });

        // We unwrap here as we want this code to exit if it fails
        window.update_with_buffer(&buffer).unwrap();
    }
    
    println!("Exiting");
}

