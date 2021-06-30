
var ul = document.getElementById("selected");
var selected = []; 
document.getElementById("add").onclick = () => {    
    let current = document.getElementById("current").value.toString();
    let found = false
    let li;
    if (current) li = document.createElement("li");
    else console.log("no disease selected");

    if (selected.length <5) {
        for (i = 0; i < selected.length; i++) {
            if (selected[i] == current) found = true;
        }
        if (!found) {
            selected.push(current);
            li.appendChild(document.createTextNode(current));
            ul.appendChild(li);
        }    
    }    
}

document.getElementById("reset").onclick = ()=> {
    selected = [];
    ul.innerHTML = "";
}

document.getElementById("submit").onclick = () => {
    if (selected.length == 5) {
        //send items of list to python from here
    }
    
}