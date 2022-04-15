const sql = require('mssql/msnodesqlv8');

// database configuration

var config =
{
    database: 'master', //databse name
    server: 'LAPTOP-UQ94PH1V', // server name
    driver: 'msnodesqlv8', // connector name

    options: {
        trustedConnection: true,
    }
};

//connect to database

sql.connect(config, function (err) {
    if (err) {
        console.log(err);
    }

    //create request object

    var request = new Request();

    //database query

    request.query('select * from userinfo', function (err, recordSet) {
        if (err) {
            console.log(err);
        }
        else {
            console.log(recordSet)
        }
    });
})
