// Simple formatting helper function for debugging
function formatDefects(content) {
    // Strip document IDs
    let formatted = content.replace(/ - Document ['"`][^'"`]+['"`]/g, '');
    formatted = formatted.replace(/ - Document [a-f0-9-]{8,}/g, '');
    
    // Format section headers
    formatted = formatted.replace(/\*\*([^*:]+):\*\*/g, '<h4 class="section-header">$1</h4>');
    
    // Format defect IDs and descriptions
    formatted = formatted.replace(/\*\*(LIFE-\d+):\*\*\s*"([^"]+)"\s*\(([^)]+)\)/g, 
        '<div class="defect-item"><strong class="defect-id">$1:</strong> "$2" <span class="severity">($3)</span></div>');
    
    // Apply severity styling
    formatted = formatted.replace(/\(Critical severity/g, '<span class="severity critical">(Critical severity');
    formatted = formatted.replace(/\(High severity/g, '<span class="severity high">(High severity');
    formatted = formatted.replace(/\(Medium severity/g, '<span class="severity medium">(Medium severity');
    formatted = formatted.replace(/\(Low severity/g, '<span class="severity low">(Low severity');
    formatted = formatted.replace(/priority\)/g, 'priority)</span>');
    
    // Format any remaining bold text
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Convert newlines to breaks
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}
