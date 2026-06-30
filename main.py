import src.preprocess as pp
import time

start_time = time.perf_counter()
chicken_images = pp.preprocess_directory("dataset/animal_counting_dataset/chicken")
end_time = time.perf_counter()
print(f"Number of chicken images: {len(chicken_images)}")
print(f"Operation took {end_time - start_time}")