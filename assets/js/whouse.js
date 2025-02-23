let quarter = 0;
let net = 0;
let stocks = 50;
const prices = [13.5, 15, 18, 15];
const transactions = document.getElementById("transactions");

function endQuarter() {
    net -= stocks; //storage cost
    const newTransaction = `<li>Spent ${stocks}$ for ${stocks} storage</li>`;
    transactions.innerHTML += newTransaction;
    quarter++;
    if(quarter == 4){
        let quarterText = "Finished";
        buttonBuy.disabled = true;
        buttonSell.disabled = true;
        if(net == 925){
            quarterText += ", Optimal 🎉";
        }else{
            quarterText += ", Suboptimal 😞";
        }

        document.getElementById("current-quarter").innerText = quarterText;
    }else{
        document.getElementById("current-quarter").innerText = `Quarter: Q${quarter + 1}`;
    }


    document.getElementById("current-net").innerText = `Money: ${net}`;
    document.getElementById("current-stocks").innerText = `Current stocks: ${stocks}`;
}

function buy(event) {
    const err = document.getElementById("error");
    err.innerText = "";

    const amount = Number(document.getElementById("amount").value);
    if(amount + stocks > 100){
        err.innerText = "You can store up to 100 stocks";
        event.preventDefault();
        return false;
    }

    const paid = (amount * prices[quarter]);
    const newTransaction = `<li>Spent ${paid}$ for buying ${amount} stocks</li>`;
    transactions.innerHTML += newTransaction;

    stocks = stocks + amount;
    net = net - paid;
    endQuarter();

    event.preventDefault();
    return false;
}

function sell(event) {
    const err = document.getElementById("error");
    err.innerText = "";

    const amount = Number(document.getElementById("amount").value);
    if(amount > stocks){
        err.innerText = "Cannot sell more than available!";
        event.preventDefault();
        return false;
    }

    stocks = stocks - amount;

    const got = (amount * prices[quarter]);
    const newTransaction = `<li>Got ${got}$ for selling ${amount} stocks</li>`;
    transactions.innerHTML += newTransaction;

    net = net + got;
    endQuarter();

    event.preventDefault();
    return false;
}

function restart(event) {
    const err = document.getElementById("error");
    err.innerText = "";

    quarter = -1;
    net = 50;
    stocks = 50;
    endQuarter();
    buttonBuy.disabled = false;
    buttonSell.disabled = false;
    transactions.innerHTML = "";

    event.preventDefault();
    return false;
}

const buttonBuy = document.getElementById("buy");
buttonBuy.addEventListener("click", buy, false);

const buttonSell = document.getElementById("sell");
buttonSell.addEventListener("click", sell, false);

const buttonRestart = document.getElementById("restart");
buttonRestart.addEventListener("click", restart, false);
