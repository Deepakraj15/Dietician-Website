import express from 'express';
import mongoose from 'mongoose';
import dotenv from 'dotenv'
import cookieParser from 'cookie-parser';
import routes from './Routes/routes.js';
dotenv.config();

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.set('view engine', 'ejs');
app.use('/', routes);
app.use(express.static('source/styles'))
app.use(express.static('source/Icons'));
app.use(express.static('source/Images'));
app.listen(5500);

// mongoose.connect(process.env.DB_CONNECTION_STRING, { useUnifiedTopology: true, useNewUrlParser: true });
// mongoose.connection.on('connected', () => {
//     console.log('Mongoose is connected');
// })
// app.listen(process.env.PORT);
