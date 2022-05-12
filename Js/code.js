
/** Login Page */

function loginCredentials() {
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    if (username == "Deepak" && password == "Deepak") {
        window.location = "/home.html";
        return false;
    }
    else {
        alert('invalid vruhh');
    }
}

/** BMI and BMR calculator */

function calculateBmi() {
    const weight = document.getElementById('weight-number').value;
    const age = document.getElementById('age').value;
    const height = document.getElementById('height-number').value;
    const gender = document.getElementById('gender-picker').value;
    const gaugeElement = document.querySelector('.gauge');
    let bmr = 0;
    let bmi = weight / ((height / 100) ** (height / 100));
    if (gender === 'male') // male
    {
        bmr = 88.3 + (13.4 * weight) + (4.7 * height) - (5.6 * age);
    }
    if (gender == 'female') //female
    {
        bmr = 47.5 + (9.2 * weight) + (3.0 * height) - (4.3 * age);
    }
    setGaugeValue(gaugeElement, bmi, bmr);


}

// setting gauge 

function setGaugeValue(gauge, value, bmr) {

    if (value <= 18.49) {
        gauge.querySelector(".gauge__fill").style.transform = `rotate(${15
            }deg)`;
        gauge.querySelector(".gauge__cover").textContent = `${Math.round(value)} Under weight`;
    }
    else if (value >= 18.5 && value <= 24.99) {

        gauge.querySelector(".gauge__fill").style.transform = `rotate(${45
            }deg)`;
        gauge.querySelector(".gauge__cover").textContent = `${Math.round(value)} Normal`;
    }
    else {
        gauge.querySelector(".gauge__fill").style.transform = `rotate(${160
            }deg)`;
        gauge.querySelector(".gauge__cover").textContent = `${Math.round(value)} Obese`;
    }
    document.querySelector(".text-result").style.visibility = 'visible';
    document.querySelector("p").innerHTML = `${Math.round(bmr)}`

}

/* drop down display */

function dropdowndisplay() {
    document.querySelector('.dropvalues').style.display = 'flex';

}

/**creating divs for scheduler */

function createDiv() {

}

/** load and reload */

// load home page

function loadCreateAcc() {
    window.location = '/newuser.html'
}

// relaod current page
function reloadPage() {
    location.reload()
}
