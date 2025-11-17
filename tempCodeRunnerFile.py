                    avg_color_int = tuple(int(np.clip(c, 0, 255)) for c in avg_color)
                    # avg_color is BGR; convert to RGB hex
                    text = "#{:02x}{:02x}{:02x}".format(avg_color_int[2], avg_color_int[1], avg_color_int[0])