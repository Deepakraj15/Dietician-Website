import express from "express"
import HomePage from "../controllers/Homepage.js";
import Login from "../controllers/Login.js";
import CalorieTracker from "../controllers/CalorieTracker.js";

const router = express.Router();

router.get('/', HomePage);
router.get('/loginUser',Login);
export default router;