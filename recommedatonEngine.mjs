//import * as tf from '@tensorflow/tfjs-node';
import { tensor } from '@tensorflow/tfjs';
const dataArray = [8, 6, 7, 5, 3, 0, 9];
const first = tensor(dataArray);
first.print();