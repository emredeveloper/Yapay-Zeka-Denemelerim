// Agent chat için conversation history
let conversationHistory = [];

// Sayfa yüklendiğinde harita oluştur ve agent durumunu kontrol et
document.addEventListener('DOMContentLoaded', function() {
    createMap();
    checkAgentHealth();
    // Şehir checkbox'larına change event listener ekle
    document.querySelectorAll('.city-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            // Otomatik harita güncelleme (opsiyonel)
        });
    });
});

function getSelectedCities() {
    const checkboxes = document.querySelectorAll('.city-checkbox:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

function createMap() {
    const selectedCities = getSelectedCities();
    const showBoundaries = document.getElementById('show-boundaries').checked;
    const showClustering = document.getElementById('show-clustering').checked;
    const mapStyle = document.getElementById('map-style').value;
    const showLines = document.getElementById('show-lines').checked;
    const statusDiv = document.getElementById('map-status');
    
    if (selectedCities.length === 0) {
        statusDiv.textContent = '⚠️ Lütfen en az bir şehir seçin!';
        statusDiv.className = 'badge bg-warning text-dark';
        statusDiv.style.display = 'inline-block';
        return;
    }
    
    statusDiv.textContent = '⏳ Harita oluşturuluyor...';
    statusDiv.className = 'badge bg-info text-white';
    statusDiv.style.display = 'inline-block';
    
    fetch('/api/create_map', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            cities: selectedCities,
            show_boundaries: showBoundaries,
            show_clustering: showClustering,
            map_style: mapStyle,
            show_lines: showLines
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('map').innerHTML = data.map_html;
            statusDiv.textContent = `✅ ${selectedCities.length} şehir gösteriliyor`;
            statusDiv.className = 'badge bg-success text-white';
            statusDiv.style.display = 'inline-block';
        } else {
            statusDiv.textContent = '❌ ' + (data.error || 'Bilinmeyen hata');
            statusDiv.className = 'badge bg-danger text-white';
            statusDiv.style.display = 'inline-block';
        }
    })
    .catch(error => {
        statusDiv.textContent = '❌ ' + error.message;
        statusDiv.className = 'badge bg-danger text-white';
        statusDiv.style.display = 'inline-block';
    });
}

function loadCityInfo() {
    const cityName = document.getElementById('city-selector').value;
    const infoDiv = document.getElementById('city-info');
    
    infoDiv.innerHTML = '<p class="text-muted">Yükleniyor...</p>';
    
    fetch(`/api/city_info/${cityName}`)
        .then(response => response.json())
        .then(data => {
            const cityData = data.data;
            const wiki = data.wikipedia;
            
            let html = `
                <h6>${data.city}</h6>
                <hr>
                <p><strong>Bölge:</strong> ${cityData.bolge}</p>
                <p><strong>Nüfus:</strong> ${cityData.nufus.toLocaleString('tr-TR')}</p>
                <p><strong>Alan:</strong> ${cityData.alan_km2.toLocaleString('tr-TR')} km²</p>
                <p><strong>Yükseklik:</strong> ${cityData.yukseklik_m} m</p>
                <p><strong>Plaka:</strong> ${String(cityData.plaka).padStart(2, '0')}</p>
                <p><strong>Koordinat:</strong> ${cityData.koordinat[0].toFixed(4)}, ${cityData.koordinat[1].toFixed(4)}</p>
                <hr>
                <h6>Wikipedia</h6>
                <p class="small">${wiki.ozet}</p>
                ${wiki.url ? `<a href="${wiki.url}" target="_blank" class="btn btn-sm btn-outline-primary">Daha Fazla</a>` : ''}
            `;
            
            infoDiv.innerHTML = html;
        })
        .catch(error => {
            infoDiv.innerHTML = '<p class="text-danger">Bilgi yüklenemedi: ' + error.message + '</p>';
        });
}

function loadStatistics() {
    const selectedCities = getSelectedCities();
    const statsDiv = document.getElementById('statistics');
    
    if (selectedCities.length === 0) {
        statsDiv.innerHTML = '<p class="text-danger">Lütfen şehir seçin!</p>';
        return;
    }
    
    statsDiv.innerHTML = '<p class="text-muted">Hesaplanıyor...</p>';
    
    fetch('/api/statistics', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ cities: selectedCities })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            statsDiv.innerHTML = '<p class="text-danger">' + data.error + '</p>';
            return;
        }
        
        if (data.single_city) {
            statsDiv.innerHTML = `
                <h6>${data.city}</h6>
                <p><strong>Nüfus:</strong> ${data.data.nufus.toLocaleString('tr-TR')}</p>
                <p><strong>Alan:</strong> ${data.data.alan_km2.toLocaleString('tr-TR')} km²</p>
                <p><strong>Bölge:</strong> ${data.data.bolge}</p>
            `;
        } else {
            statsDiv.innerHTML = `
                <p><strong>Seçilen Şehir:</strong> ${data.city_count}</p>
                <hr>
                <p><strong>En Uzak:</strong> ${data.max_distance.city1} ↔ ${data.max_distance.city2}<br>
                <small>${data.max_distance.distance.toFixed(2)} km</small></p>
                <p><strong>En Yakın:</strong> ${data.min_distance.city1} ↔ ${data.min_distance.city2}<br>
                <small>${data.min_distance.distance.toFixed(2)} km</small></p>
                <p><strong>Ortalama Mesafe:</strong> ${data.avg_distance.toFixed(2)} km</p>
                <hr>
                <p><strong>Toplam Nüfus:</strong> ${data.total_population.toLocaleString('tr-TR')}</p>
                <p><strong>En Kalabalık:</strong> ${data.max_pop_city.name}<br>
                <small>${data.max_pop_city.population.toLocaleString('tr-TR')}</small></p>
                <p><strong>Toplam Alan:</strong> ${data.total_area.toLocaleString('tr-TR')} km²</p>
            `;
        }
    })
    .catch(error => {
        statsDiv.innerHTML = '<p class="text-danger">Hata: ' + error.message + '</p>';
    });
}

function loadDistances() {
    const selectedCities = getSelectedCities();
    const distancesDiv = document.getElementById('distances');
    
    if (selectedCities.length < 2) {
        distancesDiv.innerHTML = '<p class="text-danger">En az 2 şehir seçmelisiniz!</p>';
        return;
    }
    
    distancesDiv.innerHTML = '<p class="text-muted">Hesaplanıyor...</p>';
    
    fetch('/api/distances', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ cities: selectedCities })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            distancesDiv.innerHTML = '<p class="text-danger">' + data.error + '</p>';
            return;
        }
        
        if (data.distances.length === 0) {
            distancesDiv.innerHTML = '<p class="text-muted">Mesafe bulunamadı.</p>';
            return;
        }
        
        let html = '<table class="table table-sm table-striped">';
        html += '<thead><tr><th>Şehir 1</th><th>Şehir 2</th><th>Mesafe (km)</th></tr></thead><tbody>';
        
        data.distances.forEach(d => {
            html += `<tr><td>${d.city1}</td><td>${d.city2}</td><td>${d.distance.toFixed(2)}</td></tr>`;
        });
        
        html += '</tbody></table>';
        distancesDiv.innerHTML = html;
    })
    .catch(error => {
        distancesDiv.innerHTML = '<p class="text-danger">Hata: ' + error.message + '</p>';
    });
}

function loadCitiesTable() {
    const selectedCities = getSelectedCities();
    const tableDiv = document.getElementById('cities-table');
    
    if (selectedCities.length === 0) {
        tableDiv.innerHTML = '<p class="text-danger">Lütfen şehir seçin!</p>';
        return;
    }
    
    tableDiv.innerHTML = '<p class="text-muted">Yükleniyor...</p>';
    
    fetch('/api/cities_table', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ cities: selectedCities })
    })
    .then(response => response.json())
    .then(data => {
        if (data.cities.length === 0) {
            tableDiv.innerHTML = '<p class="text-muted">Veri bulunamadı.</p>';
            return;
        }
        
        let html = '<table class="table table-sm table-striped">';
        html += '<thead><tr>';
        
        // Başlıklar
        Object.keys(data.cities[0]).forEach(key => {
            html += `<th>${key}</th>`;
        });
        
        html += '</tr></thead><tbody>';
        
        // Satırlar
        data.cities.forEach(city => {
            html += '<tr>';
            Object.values(city).forEach(value => {
                html += `<td>${value}</td>`;
            });
            html += '</tr>';
        });
        
        html += '</tbody></table>';
        tableDiv.innerHTML = html;
    })
    .catch(error => {
        tableDiv.innerHTML = '<p class="text-danger">Hata: ' + error.message + '</p>';
    });
}

// Tab değiştiğinde içeriği güncelle
document.addEventListener('DOMContentLoaded', function() {
    // Tab değiştiğinde
    const statsTab = document.getElementById('stats-tab');
    const cityTab = document.getElementById('city-tab');
    
    if (statsTab) {
        statsTab.addEventListener('shown.bs.tab', function() {
            // İstatistikler tab'ına geçildiğinde
            const statistics = document.getElementById('statistics').innerHTML.trim();
            if (!statistics || statistics.includes('Şehir seçip')) {
                // İçerik yoksa yükleme yapma
            }
        });
    }
    
    if (cityTab) {
        cityTab.addEventListener('shown.bs.tab', function() {
            // Şehir tab'ına geçildiğinde bilgi yükle
            const cityInfo = document.getElementById('city-info').innerHTML.trim();
            if (!cityInfo || cityInfo.includes('şehir seçip')) {
                loadCityInfo();
            }
        });
    }
});

// ==================== Agent Chat Functions ====================

function checkAgentHealth() {
    const statusElement = document.getElementById('agent-status-badge');
    if (!statusElement) return;
    
    statusElement.innerHTML = '<i class="bi bi-circle-fill text-secondary"></i> Durum kontrol ediliyor...';
    
    fetch('/api/agent/health')
        .then(response => response.json())
        .then(data => {
            if (data.available) {
                statusElement.innerHTML = `<i class="bi bi-circle-fill text-success"></i> Agent hazır`;
                statusElement.className = 'badge bg-light text-dark';
                statusElement.title = `${data.model_name} - ${data.tools_count} tool mevcut`;
            } else {
                statusElement.innerHTML = `<i class="bi bi-circle-fill text-warning"></i> Agent hazır değil`;
                statusElement.className = 'badge bg-light text-dark';
                statusElement.title = data.message || 'LM Studio çalıştırılıyor mu?';
            }
        })
        .catch(error => {
            console.error('Health check hatası:', error);
            statusElement.innerHTML = `<i class="bi bi-circle-fill text-danger"></i> Bağlantı hatası`;
            statusElement.className = 'badge bg-light text-dark';
            statusElement.title = error.message || 'Bağlantı hatası';
        });
}

// Şehir arama fonksiyonu
function filterCities() {
    const searchTerm = document.getElementById('city-search').value.toLowerCase();
    const cityItems = document.querySelectorAll('.city-item');
    
    cityItems.forEach(item => {
        const label = item.querySelector('label').textContent.toLowerCase();
        if (label.includes(searchTerm)) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });
}

// Tüm şehirleri seç
function selectAllCities() {
    document.querySelectorAll('.city-checkbox').forEach(checkbox => {
        checkbox.checked = true;
    });
}

// Tüm şehir seçimlerini temizle
function deselectAllCities() {
    document.querySelectorAll('.city-checkbox').forEach(checkbox => {
        checkbox.checked = false;
    });
}

function setExampleQuery(query) {
    document.getElementById('chat-input').value = query;
    sendAgentMessage();
}

// Harita ile ilgili soruları tespit et
function isMapRelatedQuestion(message) {
    const mapKeywords = ['haritada', 'harita', 'map', 'görünüyor', 'göster', 'ne var', 'neler var', 
                        'hangileri', 'hangi şehirler', 'şehirler', 'bölgeler', 'coğrafi'];
    const lowerMessage = message.toLowerCase();
    return mapKeywords.some(keyword => lowerMessage.includes(keyword));
}

// Harita screenshot'ı al - geliştirilmiş versiyon
async function captureMapScreenshot() {
    try {
        const mapContainer = document.getElementById('map');
        if (!mapContainer) {
            console.warn('Harita container bulunamadı');
            return null;
        }
        
        // html2canvas kontrolü
        if (typeof html2canvas === 'undefined') {
            console.error('html2canvas yüklü değil');
            return null;
        }
        
        // Folium haritası iframe içinde render edilir
        const iframe = mapContainer.querySelector('iframe');
        
        if (iframe) {
            try {
                // iframe içeriğine erişmeyi dene (data URI veya blob URL'ler için çalışabilir)
                const iframeSrc = iframe.src;
                console.log('[Screenshot] iframe src:', iframeSrc.substring(0, 50));
                
                // Eğer iframe src data URI veya blob URL ise, içeriğe erişebiliriz
                if (iframeSrc.startsWith('data:') || iframeSrc.startsWith('blob:')) {
                    try {
                        const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                        
                        if (iframeDoc && iframeDoc.body) {
                            // iframe içeriğini screenshot al
                            const canvas = await html2canvas(iframeDoc.body, {
                                allowTaint: true,
                                useCORS: true,
                                backgroundColor: '#ffffff',
                                scale: 0.8, // Daha hızlı için scale düşür
                                logging: false,
                                width: iframe.offsetWidth,
                                height: iframe.offsetHeight
                            });
                            
                            const dataUrl = canvas.toDataURL('image/png');
                            console.log('[Screenshot] iframe içeriği yakalandı, boyut:', dataUrl.length);
                            return dataUrl;
                        }
                    } catch (iframeAccessError) {
                        console.warn('[Screenshot] iframe içeriğine erişilemedi:', iframeAccessError.message);
                    }
                } else {
                    console.warn('[Screenshot] iframe cross-origin, içeriğe erişilemiyor');
                }
                
                // iframe'e erişilemezse, harita container'ın kendisini al
                // Ancak iframe görünümünü de eklemeye çalış
                const canvas = await html2canvas(mapContainer, {
                    allowTaint: true,
                    useCORS: true,
                    backgroundColor: '#ffffff',
                    scale: 0.8,
                    logging: false,
                    foreignObjectRendering: true
                });
                
                const dataUrl = canvas.toDataURL('image/png');
                console.log('[Screenshot] Container yakalandı, boyut:', dataUrl.length);
                return dataUrl;
                
            } catch (error) {
                console.error('[Screenshot] iframe işleme hatası:', error);
            }
        }
        
        // iframe yoksa, container'ı direkt al
        const canvas = await html2canvas(mapContainer, {
            allowTaint: true,
            useCORS: true,
            backgroundColor: '#ffffff',
            scale: 0.8,
            logging: false
        });
        
        return canvas.toDataURL('image/png');
        
    } catch (error) {
        console.error('[Screenshot] Genel hata:', error);
        return null;
    }
}

// Script yükleme helper
function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

function sendAgentMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (!message) {
        return;
    }
    
    // Kullanıcı mesajını ekle
    addChatMessage('user', message);
    input.value = '';
    
    // Loading indicator
    const loadingId = addChatMessage('assistant', 'Düşünüyorum...', true);
    
    // Harita ile ilgili soru mu kontrol et
    const isMapQuestion = isMapRelatedQuestion(message);
    const selectedCities = getSelectedCities();
    
    // Eğer harita ile ilgili bir soru ise VE şehir seçiliyse, backend'ten görsel al
    if (isMapQuestion && selectedCities.length > 0) {
        console.log('[Agent Chat] Harita sorusu tespit edildi, backend\'ten harita görseli alınıyor...');
        
        // Backend'ten harita görselini al
        fetch('/api/map/image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                cities: selectedCities,
                show_boundaries: document.getElementById('show-boundaries').checked,
                show_lines: document.getElementById('show-lines').checked,
                map_style: document.getElementById('map-style').value
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success && data.image_base64) {
                console.log('[Agent Chat] Harita görseli alındı, VLM ile analiz ediliyor...');
                sendMapScreenshotAnalysis(message, data.image_base64, loadingId);
            } else {
                console.log('[Agent Chat] Harita görseli alınamadı, backend durumu kullanılıyor...');
                sendNormalAgentMessage(message, loadingId);
            }
        })
        .catch(error => {
            console.warn('[Agent Chat] Harita görseli hatası:', error);
            sendNormalAgentMessage(message, loadingId);
        });
    } else {
        // Normal chat veya harita sorusu ama şehir yok
        sendNormalAgentMessage(message, loadingId);
    }
}

function sendNormalAgentMessage(message, loadingId) {
    // Harita durumunu al
    const mapState = {
        cities: getSelectedCities(),
        map_style: document.getElementById('map-style').value,
        show_boundaries: document.getElementById('show-boundaries').checked,
        show_clustering: document.getElementById('show-clustering').checked,
        show_lines: document.getElementById('show-lines').checked
    };
    
    // API çağrısı
    fetch('/api/agent/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: message,
            history: conversationHistory,
            map_state: mapState
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || `HTTP ${response.status}: ${response.statusText}`);
            });
        }
        return response.json();
    })
    .then(data => {
        // Loading mesajını kaldır
        removeChatMessage(loadingId);
        
        if (data.success) {
            // Assistant yanıtını ekle (formatlanmış)
            const formattedResponse = data.response.replace(/\n/g, '<br>');
            addChatMessage('assistant', formattedResponse);
            
            // Conversation history'ye ekle
            conversationHistory.push({ role: 'user', content: message });
            conversationHistory.push({ role: 'assistant', content: data.response });
            
            // History'yi son 20 mesajla sınırla
            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }
        } else {
            const errorMsg = data.error || 'Bilinmeyen hata';
            const suggestion = data.suggestion || '';
            addChatMessage('error', `Hata: ${errorMsg}${suggestion ? '<br><small>' + suggestion + '</small>' : ''}`);
        }
    })
    .catch(error => {
        removeChatMessage(loadingId);
        console.error('Agent chat hatası:', error);
        addChatMessage('error', 'Bağlantı hatası: ' + error.message);
    });
}

function sendMapScreenshotAnalysis(message, screenshot, loadingId) {
    // Harita screenshot'ı ile analiz et
    fetch('/api/map/screenshot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image_base64: screenshot,
            question: message
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || `HTTP ${response.status}: ${response.statusText}`);
            });
        }
        return response.json();
    })
    .then(data => {
        // Loading mesajını kaldır
        removeChatMessage(loadingId);
        
        if (data.success) {
            // Assistant yanıtını ekle (formatlanmış)
            const responseText = data.analysis || data.response;
            const formattedResponse = responseText.replace(/\n/g, '<br>');
            addChatMessage('assistant', formattedResponse);
            
            // Conversation history'ye ekle
            conversationHistory.push({ role: 'user', content: message });
            conversationHistory.push({ role: 'assistant', content: responseText });
            
            // History'yi son 20 mesajla sınırla
            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }
        } else {
            const errorMsg = data.error || 'Bilinmeyen hata';
            addChatMessage('error', `Hata: ${errorMsg}`);
        }
    })
    .catch(error => {
        removeChatMessage(loadingId);
        console.error('Harita screenshot analiz hatası:', error);
        // Hata durumunda normal chat'i dene
        sendNormalAgentMessage(message, loadingId);
    });
}

function addChatMessage(role, content, isLoading = false) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageId = 'msg-' + Date.now();
    
    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = 'mb-2';
    
    let badgeClass = 'bg-secondary';
    let icon = '👤';
    
    if (role === 'assistant') {
        badgeClass = 'bg-primary';
        icon = '🤖';
    } else if (role === 'error') {
        badgeClass = 'bg-danger';
        icon = '⚠️';
    }
    
    if (isLoading) {
        messageDiv.innerHTML = `
            <div class="d-flex align-items-center">
                <span class="badge ${badgeClass} me-2">${icon}</span>
                <div class="spinner-border spinner-border-sm me-2" role="status">
                    <span class="visually-hidden">Yükleniyor...</span>
                </div>
                <span class="text-muted">${content}</span>
            </div>
        `;
    } else {
        const contentHtml = content.replace(/\n/g, '<br>');
        messageDiv.innerHTML = `
            <div class="d-flex align-items-start">
                <span class="badge ${badgeClass} me-2 mt-1">${icon}</span>
                <div class="flex-grow-1">
                    <div class="small">${contentHtml}</div>
                </div>
            </div>
        `;
    }
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    return messageId;
}

function removeChatMessage(messageId) {
    const messageDiv = document.getElementById(messageId);
    if (messageDiv) {
        messageDiv.remove();
    }
}

// ==================== Image Analysis Functions ====================

function analyzeImage() {
    const imageInput = document.getElementById('image-input');
    const questionInput = document.getElementById('image-question');
    const file = imageInput.files[0];
    
    if (!file) {
        alert('Lütfen bir görsel dosyası seçin!');
        return;
    }
    
    // Dosya boyutu kontrolü (16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('Dosya boyutu 16MB\'dan büyük olamaz!');
        return;
    }
    
    // Kullanıcı mesajını ekle
    const question = questionInput.value.trim() || 'Bu görseli analiz et.';
    addChatMessage('user', `📷 Görsel Analiz: ${question}`);
    
    // Loading indicator
    const loadingId = addChatMessage('assistant', 'Görsel analiz ediliyor...', true);
    
    // FormData oluştur
    const formData = new FormData();
    formData.append('image', file);
    formData.append('question', question);
    
    // API çağrısı
    fetch('/api/agent/analyze_image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Loading mesajını kaldır
        removeChatMessage(loadingId);
        
        if (data.success) {
            // Görsel önizleme ekle
            const imageUrl = `/static/uploads/${data.image_filename}`;
            addImageMessage('assistant', data.response, imageUrl);
            
            // Conversation history'ye ekle
            conversationHistory.push({ 
                role: 'user', 
                content: `Görsel analiz: ${question}` 
            });
            conversationHistory.push({ 
                role: 'assistant', 
                content: data.response 
            });
        } else {
            addChatMessage('error', 'Hata: ' + (data.error || 'Bilinmeyen hata'));
        }
        
        // Input'ları temizle
        imageInput.value = '';
        questionInput.value = '';
    })
    .catch(error => {
        removeChatMessage(loadingId);
        addChatMessage('error', 'Bağlantı hatası: ' + error.message);
    });
}

function addImageMessage(role, content, imageUrl) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageId = 'msg-' + Date.now();
    
    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = 'mb-2';
    
    const contentHtml = content.replace(/\n/g, '<br>');
    messageDiv.innerHTML = `
        <div class="d-flex align-items-start">
            <span class="badge bg-primary me-2 mt-1">🤖</span>
            <div class="flex-grow-1">
                <div class="mb-2">
                    <img src="${imageUrl}" alt="Analiz edilen görsel" 
                         class="img-thumbnail" style="max-width: 300px; cursor: pointer;" 
                         onclick="window.open('${imageUrl}', '_blank')">
                </div>
                <div class="small">${contentHtml}</div>
            </div>
        </div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    return messageId;
}

// Drag & Drop için görsel yükleme
document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('image-input');
    const chatMessages = document.getElementById('chat-messages');
    
    // Drag & Drop event listeners
    if (chatMessages) {
        chatMessages.addEventListener('dragover', function(e) {
            e.preventDefault();
            chatMessages.style.backgroundColor = '#e3f2fd';
        });
        
        chatMessages.addEventListener('dragleave', function(e) {
            e.preventDefault();
            chatMessages.style.backgroundColor = '#f8f9fa';
        });
        
        chatMessages.addEventListener('drop', function(e) {
            e.preventDefault();
            chatMessages.style.backgroundColor = '#f8f9fa';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    imageInput.files = files;
                    analyzeImage();
                } else {
                    alert('Lütfen bir görsel dosyası seçin!');
                }
            }
        });
    }
});

