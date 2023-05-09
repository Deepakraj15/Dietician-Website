import express from "express"
import HomePage from "../controllers/Homepage.js";
import Login from "../controllers/Login.js";
import Calorie from "../controllers/Calorie.js";
import FoodRecommender from "../controllers/FoodRecommender.js";
import Video from "../controllers/video.js";
import newUser from "../controllers/newUser.js";
import addNewUser from "../controllers/addNewUser.js";
import FoodRecommenderIndex from "../controllers/FoodRecommenderIndex.js";
import addCalorie from "../controllers/addCalorie.js";

const router = express.Router();

router.get('/', HomePage);
router.get('/loginUser', Login);
router.get('/videos', Video);
router.get('/diet', FoodRecommenderIndex);
router.get('/foodrecommendations', FoodRecommender);
router.get('/signup', newUser);
router.get('/createNewUser', addNewUser);
router.get('/calorietracker', Calorie);
router.get('/addData', addCalorie);
export default router;