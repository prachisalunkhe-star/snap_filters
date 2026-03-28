import os
from PIL import Image, ImageDraw

def create_placeholders():
    path = "filters/assets"
    os.makedirs(path, exist_ok=True)
    
    assets = [
        ("sunglasses.png", (0, 0, 0, 200), "rect"),
        ("dog_ears.png", (139, 69, 19, 255), "ears"),
        ("cat_ears.png", (50, 50, 50, 255), "ears"),
        ("crown.png", (255, 215, 0, 255), "crown"),
        ("rainbow.png", (255, 0, 0, 150), "rainbow"),
        ("flower_crown.png", (255, 105, 180, 200), "crown"),
        ("bunny_ears.png", (255, 255, 255, 255), "ears"),
        ("fire_halo.png", (255, 69, 0, 200), "crown"),
        ("nerd_glasses.png", (0, 0, 0, 255), "rect"),
        ("devil_horns.png", (255, 0, 0, 255), "ears"),
        ("pirate_hat.png", (20, 20, 20, 255), "crown"),
        ("pixel_glasses.png", (0, 0, 0, 255), "rect"),
    ]
    
    for name, color, shape in assets:
        img = Image.new("RGBA", (500, 300), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        if shape == "rect":
            draw.rectangle([50, 100, 200, 180], fill=color)
            draw.rectangle([300, 100, 450, 180], fill=color)
            draw.line([200, 140, 300, 140], fill=color, width=10)
        elif shape == "ears":
            draw.polygon([(50, 100), (150, 0), (250, 100)], fill=color)
            draw.polygon([(300, 100), (400, 0), (500, 100)], fill=color)
        elif shape == "crown":
            draw.rectangle([100, 150, 400, 250], fill=color)
            for x in range(100, 450, 75):
                draw.polygon([(x, 150), (x+37, 50), (x+75, 150)], fill=color)
        
        img.save(os.path.join(path, name))

if __name__ == "__main__":
    create_placeholders()