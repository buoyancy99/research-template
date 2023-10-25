import numpy as np
import cv2

from experiments.exp_base import BaseExperiment
from algorithms.examples.helloworld.example_algos import ExampleAlgo, ExampleBackwardAlgo


class HelloWorldExperiment(BaseExperiment):
    compatible_algorithms = {
        "example_helloworld_1": ExampleAlgo,
        "example_helloworld_2": ExampleBackwardAlgo,
    }

    def main(self):
        """
        By default an experiment runs a `main` task, which calls this `main` method. If your experiment has multiple
        tasks, for example, a `train` stage and a `test` stage, you can define a method for each under the
        task name. They will be called automatically when the names appear in the yaml file for experiment. For example,
        `configurations/experiments/base_experiment.yaml` has its `tasks` field containing only a `main` task,
        which we defines here. So at run time it will only run this `main` method. On the other hand, if you take a look
        at the example `configurations/experiments/classification_experiment.yaml`, it has two tasks, which wil be run
        sequentially.
        """

        # most of your experiments should be here!

        message = "Hello World!"
        print(f"Original Message: {message}")

        # run selected algorithm specified in configurations/config.yaml
        formatted_message = self.algo.run(message)
        print(f"Fortmatted Message: {formatted_message}")

        # now we show logging an image to cloud, with formatted_message in the image
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        color = (255, 255, 255)
        text_size = cv2.getTextSize(formatted_message, font, font_scale, font_thickness)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2
        cv2.putText(image, formatted_message, (text_x, text_y), font, font_scale, color, font_thickness)
        self.logger.log_image("formatted_message", [image])
