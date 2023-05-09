import express from 'express';
import mongoose from 'mongoose';
import dotenv from 'dotenv'
import bodyParser from 'body-parser';
import cookieParser from 'cookie-parser';
import routes from './Routes/routes.js';
import User from './models/user.js';
import { Url } from 'url';

dotenv.config();

const app = express();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cookieParser());
app.set('view engine', 'ejs');
app.use('/', routes);
app.use(express.static('source/styles'))
app.use(express.static('source/Icons'));
app.use(express.static('source/Images'));
app.listen(8000);
app.use('/demo', async(req,res) => {
    const user = new User({
  name: 'Deepak',
  age: 20,
  gender: 'male',
        password: '12345',
  
  calorieHistory: [
    { date: 1, calories: 1500 },
    { date: 2, calories: 1800 },
      { date: 3, calories: 2000 },
      { date: 4, calories: 2500 },
    { date: 5, calories: 2060 },
  ],
  weight: 55,
  height: 172
});

try {
  const savedUser = await user.save();
  console.log('User saved successfully:', savedUser);
} catch (err) {
  console.error(err);
}

})

mongoose.connect(process.env.DB_CONNECTION_STRING, { useUnifiedTopology: true, useNewUrlParser: true });
mongoose.connection.on('connected', () => {
    console.log('Mongoose is connected');
})
app.listen(process.env.PORT);
