const mongo = require('mongodb').MongoClient
const url = 'mongodb://localhost:27017';
const dbName = 'exampleproject';
mongo.connect(url, (err, client) => {
    if (err) {
        console.error(err)
        return
    }
    console.log('Connected successfully to server')
    const db = client.db(dbName)
    //console.log(db);

    // creating collections

    const collection = db.collection('collectionname');
    insertuser(collection);
    //inserting objects
    function insertuser(collection) {
        collection.insertOne({ name: 'Test', age: '30' }, ((error, item) => {
            if (error) {
                console.error(error)

            }
            // console.log(item);
            console.log('collection is successful');
            // const result = db.collection('collectionname').find({ name: 'Test' });
            // console.log(result);
        }))
    }
    function findUser(username, password, collection) {

    }



})
